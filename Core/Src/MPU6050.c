

#include "MPU6050.h"
#include "main.h"
#include "i2c.h"
#include <stdio.h>

Struct_MPU6050 MPU6050;

static float LSB_Sensitivity_ACC;
static float LSB_Sensitivity_GYRO;

// Function to write a single byte to an MPU6050 register
void MPU6050_Writebyte(uint8_t reg_addr, uint8_t val)
{
	HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, reg_addr, I2C_MEMADD_SIZE_8BIT, &val, 1, 1);
}

// Function to write multiple bytes to MPU6050 registers
void MPU6050_Writebytes(uint8_t reg_addr, uint8_t len, uint8_t* data)
{
	HAL_I2C_Mem_Write(&hi2c1, MPU6050_ADDR, reg_addr, I2C_MEMADD_SIZE_8BIT, data, len, 1);
}

// Function to read a single byte from an MPU6050 register
void MPU6050_Readbyte(uint8_t reg_addr, uint8_t* data)
{
	HAL_I2C_Mem_Read(&hi2c1, MPU6050_ADDR, reg_addr, I2C_MEMADD_SIZE_8BIT, data, 1, 1);
}

// Function to read multiple bytes from MPU6050 registers
void MPU6050_Readbytes(uint8_t reg_addr, uint8_t len, uint8_t* data)
{
	HAL_I2C_Mem_Read(&hi2c1, MPU6050_ADDR, reg_addr, I2C_MEMADD_SIZE_8BIT, data, len, 1);
}

// Initializes the MPU6050 sensor with desired settings
void MPU6050_Initialization(void)
{
	HAL_Delay(50); // Small delay for sensor power-up
	uint8_t who_am_i = 0;
	printf("Checking MPU6050...\n");

	// Read WHO_AM_I register to verify sensor presence
	MPU6050_Readbyte(MPU6050_WHO_AM_I, &who_am_i);
	if(who_am_i == 0x68) // 0x68 is the default WHO_AM_I value for MPU6050
	{
		printf("MPU6050 who_am_i = 0x%02x...OK\n", who_am_i);
	}
	else
	{
		printf("ERROR!\n");
		printf("MPU6050 who_am_i : 0x%02x should be 0x68\n", who_am_i);
		while(1) // Loop indefinitely on error
		{
			printf("who am i error. Can not recognize mpu6050\n");
			HAL_Delay(100);
		}
	}

	// Reset the MPU6050 module
	MPU6050_Writebyte(MPU6050_PWR_MGMT_1, 0x1<<7); // Set DEVICE_RESET bit
	HAL_Delay(100); // Wait for reset to complete

	// Power Management setting: Wake up MPU6050 (clear SLEEP bit)
	MPU6050_Writebyte(MPU6050_PWR_MGMT_1, 0x00);
	HAL_Delay(50);

	// --- MPU6050 Sample Rate and DLPF Configuration for 200Hz Output ---

	// Sample Rate Divider (SMPLRT_DIV, Register 25):
	// Sample Rate = Gyroscope Output Rate / (1 + SMPLRT_DIV)
	// With DLPF_CFG = 0x01 (below), Gyro Output Rate is 1kHz.
	// To get 200Hz: 200 = 1000 / (1 + SMPLRT_DIV) => 1 + SMPLRT_DIV = 5 => SMPLRT_DIV = 4
	MPU6050_Writebyte(MPU6050_SMPRT_DIV, 0x04); // Set SMPLRT_DIV to 4
	HAL_Delay(50);

	// Configuration Register (CONFIG, Register 26): FSYNC and DLPF setting
	// Set DLPF_CFG to 0x01 for a Digital Low Pass Filter with:
	// Accel Bandwidth: 184Hz, Delay: 2.0ms
	// Gyro Bandwidth: 188Hz, Delay: 1.9ms
	MPU6050_Writebyte(MPU6050_CONFIG, 0x01); // Set DLPF_CFG to 1
	HAL_Delay(50);

	// --- End MPU6050 Sample Rate and DLPF Configuration ---


	// GYRO FULL SCALE setting (GYRO_CONFIG, Register 27)
	/* FS_SEL  Full Scale Range
	 * 0    	+-250 degree/s
	 * 1		+-500 degree/s
	 * 2		+-1000 degree/s
	 * 3		+-2000 degree/s	*/
	uint8_t FS_SCALE_GYRO = 0x3; // Setting to +/-2000 deg/s
	MPU6050_Writebyte(MPU6050_GYRO_CONFIG, FS_SCALE_GYRO<<3);
	HAL_Delay(50);

	// ACCEL FULL SCALE setting (ACCEL_CONFIG, Register 28)
	/* FS_SEL  Full Scale Range
	 * 0    	+-2g
	 * 1		+-4g
	 * 2		+-8g
	 * 3		+-16g	*/
	uint8_t FS_SCALE_ACC = 0x3; // Setting to +/-2g
	MPU6050_Writebyte(MPU6050_ACCEL_CONFIG, FS_SCALE_ACC<<3);
	HAL_Delay(50);

	// Get LSB sensitivities based on the set full-scale ranges
	MPU6050_Get_LSB_Sensitivity(FS_SCALE_GYRO, FS_SCALE_ACC);

	// Interrupt PIN setting (INT_PIN_CFG, Register 55)
	uint8_t INT_LEVEL = 0x0; // 0 - active high, 1 - active low
	uint8_t LATCH_INT_EN = 0x1; // 1 - interrupt pin latches until cleared
	uint8_t INT_RD_CLEAR = 0x1; // 1 - interrupt flag cleared by any read operation
	MPU6050_Writebyte(MPU6050_INT_PIN_CFG, (INT_LEVEL<<7)|(LATCH_INT_EN<<5)|(INT_RD_CLEAR<<4));
	HAL_Delay(50);

	// Interrupt enable setting (INT_ENABLE, Register 56)
	uint8_t DATA_RDY_EN = 0x1; // Enable Data Ready Interrupt
	MPU6050_Writebyte(MPU6050_INT_ENABLE, DATA_RDY_EN);
	HAL_Delay(50);

	printf("MPU6050 setting is finished\n");
}

/* Get Raw Data from sensor */
void MPU6050_Get6AxisRawData(Struct_MPU6050* mpu6050)
{
	uint8_t data[14];
	// Read 14 bytes starting from ACCEL_XOUT_H (0x3B)
	// This includes Accel X, Y, Z, Temperature, Gyro X, Y, Z
	MPU6050_Readbytes(MPU6050_ACCEL_XOUT_H, 14, data);

	// Combine high and low bytes to get 16-bit raw values
	mpu6050->acc_x_raw = (data[0] << 8) | data[1];
	mpu6050->acc_y_raw = (data[2] << 8) | data[3];
	mpu6050->acc_z_raw = (data[4] << 8) | data[5];

	mpu6050->temperature_raw = (data[6] << 8) | data[7];

	mpu6050->gyro_x_raw = ((data[8] << 8) | data[9]);
	mpu6050->gyro_y_raw = ((data[10] << 8) | data[11]);
	mpu6050->gyro_z_raw = ((data[12] << 8) | data[13]);
}

// Determines the LSB sensitivity for accelerometer and gyroscope based on full-scale range settings
void MPU6050_Get_LSB_Sensitivity(uint8_t FS_SCALE_GYRO, uint8_t FS_SCALE_ACC)
{
	switch(FS_SCALE_GYRO)
	{
	case 0: // +/-250 deg/s
		LSB_Sensitivity_GYRO = 131.f;
		break;
	case 1: // +/-500 deg/s
		LSB_Sensitivity_GYRO = 65.5f;
		break;
	case 2: // +/-1000 deg/s
		LSB_Sensitivity_GYRO = 32.8f;
		break;
	case 3: // +/-2000 deg/s
		LSB_Sensitivity_GYRO = 16.4f;
		break;
	}
	switch(FS_SCALE_ACC)
	{
	case 0: // +/-2g
		LSB_Sensitivity_ACC = 16384.f;
		break;
	case 1: // +/-4g
		LSB_Sensitivity_ACC = 8192.f;
		break;
	case 2: // +/-8g
		LSB_Sensitivity_ACC = 4096.f;
		break;
	case 3: // +/-16g
		LSB_Sensitivity_ACC = 2048.f;
		break;
	}
}

/* Convert Raw Data to physical units (acc_raw -> g, gyro_raw -> degree per second) */
void MPU6050_DataConvert(Struct_MPU6050* mpu6050)
{
//	mpu6050->acc_x = mpu6050->acc_x_raw / LSB_Sensitivity_ACC;
//	mpu6050->acc_y = mpu6050->acc_y_raw / LSB_Sensitivity_ACC;
//	mpu6050->acc_z = mpu6050->acc_z_raw / LSB_Sensitivity_ACC;
	mpu6050->acc_x = (mpu6050->acc_x_raw / LSB_Sensitivity_ACC) * 9.81f;
	mpu6050->acc_y = (mpu6050->acc_y_raw / LSB_Sensitivity_ACC) * 9.81f;
	mpu6050->acc_z = (mpu6050->acc_z_raw / LSB_Sensitivity_ACC) * 9.81f;

	mpu6050->temperature = (float)(mpu6050->temperature_raw)/340+36.53;

	mpu6050->gyro_x = mpu6050->gyro_x_raw / LSB_Sensitivity_GYRO;
	mpu6050->gyro_y = mpu6050->gyro_y_raw / LSB_Sensitivity_GYRO;
	mpu6050->gyro_z = mpu6050->gyro_z_raw / LSB_Sensitivity_GYRO;
}


// Checks if new data is ready from the MPU6050 by reading the interrupt pin
int MPU6050_DataReady(void)
{
	// This function directly reads the state of the MPU6050's interrupt pin.
	// The interrupt pin is configured to latch high when new data is ready and
	// is cleared upon any read operation from the sensor's data registers.
	return HAL_GPIO_ReadPin(MPU6050_INT_PORT, MPU6050_INT_PIN);
}

// Processes the MPU6050 data: reads raw values and converts them to physical units
void MPU6050_ProcessData(Struct_MPU6050* mpu6050)
{
	MPU6050_Get6AxisRawData(mpu6050);
	MPU6050_DataConvert(mpu6050);
}
