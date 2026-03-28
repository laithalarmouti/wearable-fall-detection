

///----------------------------------------------------------------------
#include "main.h"
#include "i2c.h"
#include "usart.h"
#include "gpio.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "MPU6050.h"
#include "features.h"

// TensorFlow Lite Micro
#include "../tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "../tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "../tensorflow/lite/micro/micro_interpreter.h"
#include "../tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// TFLite model
extern const unsigned char fall_detection_model_tflite[];
extern const unsigned int fall_detection_model_tflite_len;

extern "C" int _write(int file, uint8_t* ptr, int len) {
    HAL_UART_Transmit(&huart2, ptr, len, HAL_MAX_DELAY);
    return len;
}

// Constants
constexpr int kTensorArenaSize = 4 * 1024;
constexpr int kWindowSize = 256;
constexpr int kFeatureCount = 30;
constexpr float kFallThreshold = 0.95f;

// TFLite buffers
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Sensor data buffer
float window_buffer[kWindowSize][6];
uint16_t window_index = 0;
uint16_t sample_count = 0; // Tracks how many samples have been added to the buffer since reset

// Timer for window fill
static uint32_t window_fill_start_time = 0; // To measure the time to fill 256 downsampled samples


// --- Smoothing filter state ---
static float ax_prev = 0, ay_prev = 0, az_prev = 0;
static float gx_prev = 0, gy_prev = 0, gz_prev = 0;
#define ALPHA 0.1f // ALPHA = 1.0f means no smoothing, just passes raw value


float smooth(float x, float *prev) {
    *prev = ALPHA * x + (1.0f - ALPHA) * (*prev);
    return *prev;
}


// Function declarations
void SystemClock_Config(void);
void Error_Handler(void);
void reset_buffer();

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_I2C1_Init();
    MX_USART2_UART_Init();
    MPU6050_Initialization();
    reset_buffer(); //reInitialize buffer and counters

    model = tflite::GetModel(fall_detection_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        printf("Model version mismatch!\n");
        Error_Handler();
    }


    tflite::MicroMutableOpResolver<2> resolver;
    resolver.AddFullyConnected();
    resolver.AddLogistic();

    // Create a static interpreter instance
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, nullptr);
    interpreter = &static_interpreter;

    // Allocate tensors from the provided arena
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("AllocateTensors failed\n");
        Error_Handler();
    }


    input = interpreter->input(0);
    output = interpreter->output(0);

    printf("Setup complete, starting loop...\n");

    while (1) {
        // Only proceed if MPU6050 has new data available
        if (MPU6050_DataReady() != 1) continue;

        MPU6050_ProcessData(&MPU6050); // Read raw values from MPU6050

        float ax_raw = MPU6050.acc_x_raw / 16384.0f;
        float ay_raw = MPU6050.acc_y_raw / 16384.0f;
        float az_raw = MPU6050.acc_z_raw / 16384.0f;
        float gx_raw = MPU6050.gyro_x_raw / 131.0f;
        float gy_raw = MPU6050.gyro_y_raw / 131.0f;
        float gz_raw = MPU6050.gyro_z_raw / 131.0f;

        // Apply smoothing to the raw values (ALPHA = 1.0f means no actual smoothing)
        float ax = smooth(ax_raw, &ax_prev);
        float ay = smooth(ay_raw, &ay_prev);
        float az = smooth(az_raw, &az_prev);
        float gx = smooth(gx_raw, &gx_prev);
        float gy = smooth(gy_raw, &gy_prev);
        float gz = smooth(gz_raw, &gz_prev);

        // Check for invalid sensor values (NaN/INF)
        if (!isfinite(ax) || !isfinite(ay) || !isfinite(az) ||
            !isfinite(gx) || !isfinite(gy) || !isfinite(gz)) {
            printf("⚠️ Skipping frame: invalid sensor values\n");
            continue;
        }


        // Start timer when the first sample of a new window is added
        if (sample_count == 0 && window_index == 0) {
            window_fill_start_time = HAL_GetTick();
        }

        // Store the data into the circular window_buffer
        window_buffer[window_index][0] = ax;
        window_buffer[window_index][1] = ay;
        window_buffer[window_index][2] = az;
        window_buffer[window_index][3] = gx;
        window_buffer[window_index][4] = gy;
        window_buffer[window_index][5] = gz;

        // Increment window_index using modulo to handle circular buffer
        window_index = (window_index + 1) % kWindowSize;
        sample_count++; // Increment sample_count for each sample added to the buffer


        if (sample_count >= kWindowSize) {
            uint32_t elapsed_time = HAL_GetTick() - window_fill_start_time;
            printf("⏱️ Filled %d samples in %lu ms (at 200Hz)\n", kWindowSize, elapsed_time);


            float aligned_window[kWindowSize][6];
            int start = window_index;
            for (int i = 0; i < kWindowSize; i++) {
                int idx = (start + i) % kWindowSize;
                for (int j = 0; j < 6; j++) {
                    aligned_window[i][j] = window_buffer[idx][j];
                }
            }


            bool has_stale = false;
            for (int i = 0; i < kWindowSize && !has_stale; i++) {
                for (int j = 0; j < 6; j++) {
                    if (aligned_window[i][j] == 0.0f) {
                        has_stale = true;
                        break;
                    }
                }
            }
            if (has_stale) {
                printf(" Skipping inference due to stale values in buffer (initial fill/reset?)\n");
                // Reset buffer to ensure a clean start for the next window
                reset_buffer();
                continue; //
            }

            // Extract features from the aligned window data
            int8_t features_quantized[kFeatureCount];
            extract_top_features(aligned_window, kWindowSize, features_quantized);

            printf("🔍 Features (quantized): ");
            for (int i = 0; i < kFeatureCount; i++) {
                printf("f%d=%d ", i, features_quantized[i]);
            }
            printf("\n");

            // Copy quantized features directly to input tensor
            for (int i = 0; i < kFeatureCount; i++) {
                input->data.int8[i] = features_quantized[i];
            }


            uint32_t inference_start = HAL_GetTick();
            // Run inference
            if (interpreter->Invoke() != kTfLiteOk) {
                printf("Inference failed\n");
                Error_Handler();
            }
            uint32_t inference_end = HAL_GetTick();
            uint32_t inference_duration = inference_end - inference_start;

            printf("🕒 Inference took %lu ms\n", inference_duration);

            // Get the raw quantized prediction and dequantize it
            int8_t raw_pred = output->data.int8[0];
            float prediction = (raw_pred - output->params.zero_point) * output->params.scale;



            printf("Prediction = %.3f (raw %d) => %s\n", prediction, raw_pred,
                   prediction > kFallThreshold ? "FALL" : "NO FALL");

            // If a fall is detected, activate the LED and reset the buffer
            if (prediction > kFallThreshold) {
                HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET); // Turn on LED
                HAL_Delay(500);
                HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET); // Turn off LED
                printf(" FALL DETECTED! Resetting buffer...\n");
                reset_buffer(); // Reset for next detection cycle
                HAL_Delay(1000); // Small delay
            }


            sample_count = 0;

        }
    }
}

// Resets
void reset_buffer() {
    for (int i = 0; i < kWindowSize; i++) {
        for (int j = 0; j < 6; j++) {
            window_buffer[i][j] = 0.0f;
        }
    }
    sample_count = 0;
    window_index = 0;
    window_fill_start_time = 0;

    ax_prev = ay_prev = az_prev = 0;
    gx_prev = gy_prev = gz_prev = 0;
}

// System Clock Configuration
void SystemClock_Config(void) {
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PLLM = 16;
    RCC_OscInitStruct.PLL.PLLN = 336;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
    RCC_OscInitStruct.PLL.PLLQ = 4;

    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                    | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
      RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
       RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
        RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
       RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK) {
        Error_Handler();
    }
}

// Error Handler
void Error_Handler(void) {
    __disable_irq();
    while (1) {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET); // Turn on LED to indicate error
        HAL_Delay(300);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET); // Turn off LED
        HAL_Delay(300);
    }
}



