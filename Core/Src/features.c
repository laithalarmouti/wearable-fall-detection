

//----------------------------------WORKING
//FEATURE REWORK

#include "features.h"
#include <math.h>
#include <string.h>
#include "arm_math.h"
#include "arm_const_structs.h"

#include "arm_sorting.h"



#define EPSILON 1e-8
#define NUM_BINS 10

static float mean(const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i];
    return sum / n;
}

static float stddev(const float *x, int n, float m) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += (x[i] - m) * (x[i] - m);
    return sqrtf(s / (n - 1)); // unbiased
}

static float var(const float *x, int n, float m) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += (x[i] - m) * (x[i] - m);
    return s / (n - 1);
}

static float min_val(const float *x, int n) {
    float minv = x[0];
    for (int i = 1; i < n; ++i)
        if (x[i] < minv) minv = x[i];
    return minv;
}

static float max_val(const float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; ++i)
        if (x[i] > maxv) maxv = x[i];
    return maxv;
}

static float median(float *x, int n) {
    float temp[n]; memcpy(temp, x, n * sizeof(float));
    float sorted[n];

    arm_sort_instance_f32 S;
    arm_sort_init_f32(&S, ARM_SORT_BUBBLE, 1);
    arm_sort_f32(&S, temp, sorted, n);

    return n % 2 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0f;
}

static float percentile(float *x, int n, float p) {
    float temp[n]; memcpy(temp, x, n * sizeof(float));
    float sorted[n];

    arm_sort_instance_f32 S;
    arm_sort_init_f32(&S, ARM_SORT_BUBBLE, 1);
    arm_sort_f32(&S, temp, sorted, n);

    float idx = p * (n - 1);
    int i = (int)idx;
    float frac = idx - i;
    return sorted[i] * (1.0f - frac) + sorted[i + 1] * frac;
}


static float iqr(float *x, int n) {
    return percentile(x, n, 0.75f) - percentile(x, n, 0.25f);
}

static float rms(const float *x, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += x[i] * x[i];
    return sqrtf(s / n);
}

static float waveform_length(const float *x, int n) {
    float wl = 0.0f;
    for (int i = 1; i < n; ++i) wl += fabsf(x[i] - x[i - 1]);
    return wl;
}

static float zero_crossing_rate(const float *x, int n) {
    int count = 0;
    for (int i = 1; i < n; ++i)
        if ((x[i - 1] >= 0 && x[i] < 0) || (x[i - 1] < 0 && x[i] >= 0)) count++;
    return (float)count / n;
}

static float mean_crossing_rate(const float *x, int n) {
    float m = mean(x, n);
    int count = 0;
    for (int i = 1; i < n; ++i)
        if ((x[i - 1] - m) * (x[i] - m) < 0) count++;
    return (float)count / n;
}

static float entropy_hist(const float *x, int n) {
    float minv = min_val(x, n), maxv = max_val(x, n), range = maxv - minv;
    if (range < EPSILON) return 0.0f;
    int bins[NUM_BINS] = {0};
    for (int i = 0; i < n; ++i) {
        int bin = (int)(NUM_BINS * (x[i] - minv) / range);
        if (bin >= NUM_BINS) bin = NUM_BINS - 1;
        bins[bin]++;
    }
    float ent = 0.0f;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (bins[i]) {
            float p = (float)bins[i] / n;
            ent -= p * logf(p);
        }
    }
    return ent;
}

void extract_features(float window[][6], int window_size, float features[294])  {
    int base = 0;
    for (int axis = 0; axis < 6; ++axis) {

    	float sig[window_size];
    	for (int i = 0; i < window_size; ++i)
    	    sig[i] = window[i][axis];

        float m = mean(sig, window_size);
        float s = stddev(sig, window_size, m);

        features[base++] = m;
        features[base++] = s;
        features[base++] = var(sig, window_size, m);
        features[base++] = min_val(sig, window_size);
        features[base++] = max_val(sig, window_size);
        features[base++] = median((float*)sig, window_size);
        features[base++] = percentile((float*)sig, window_size, 0.25f);
        features[base++] = percentile((float*)sig, window_size, 0.75f);
        features[base++] = max_val(sig, window_size) - min_val(sig, window_size);

        float sk = 0.0f, ku = 0.0f;
        for (int i = 0; i < window_size; ++i) {
            float d = sig[i] - m;
            sk += d * d * d;
            ku += d * d * d * d;
        }
        sk /= (window_size * s * s * s + EPSILON);
        ku = ku / (window_size * s * s * s * s + EPSILON) - 3;
        features[base++] = sk;
        features[base++] = ku;

        features[base++] = rms(sig, window_size);
        float energy = 0.0f;
        for (int i = 0; i < window_size; ++i) energy += sig[i] * sig[i];
        features[base++] = energy;

        float mad = 0.0f;
        for (int i = 0; i < window_size; ++i) mad += fabsf(sig[i] - m);
        features[base++] = mad / window_size;

        features[base++] = waveform_length(sig, window_size);
        features[base++] = zero_crossing_rate(sig, window_size);
        features[base++] = mean_crossing_rate(sig, window_size);
        features[base++] = iqr((float*)sig, window_size);
        features[base++] = waveform_length(sig, window_size);
        features[base++] = window_size;
        features[base++] = sqrtf(energy);
        features[base++] = entropy_hist(sig, window_size);

        // FFT features
        float fft_buf[window_size];
        memcpy(fft_buf, sig, sizeof(float) * window_size);
        arm_rfft_fast_instance_f32 S;
        arm_rfft_fast_init_f32(&S, window_size);
        arm_rfft_fast_f32(&S, fft_buf, fft_buf, 0);

        float fft_mag[window_size / 2];
        for (int i = 0; i < window_size / 2; ++i) {
            float real = fft_buf[2 * i];
            float imag = fft_buf[2 * i + 1];
            fft_mag[i] = sqrtf(real * real + imag * imag) * window_size;
        }

        float fft_mean = mean(fft_mag, window_size / 2);
        float fft_std = stddev(fft_mag, window_size / 2, fft_mean);
        float fft_max = max_val(fft_mag, window_size / 2);
        float fft_sum = 0.0f;
        for (int i = 0; i < window_size / 2; ++i) fft_sum += fft_mag[i];

        float diff[window_size / 2 - 1];
        for (int i = 0; i < window_size / 2 - 1; ++i) diff[i] = fft_mag[i + 1] - fft_mag[i];
        float diff_mean = mean(diff, window_size / 2 - 1);
        float diff_std = stddev(diff, window_size / 2 - 1, diff_mean);
        float diff_max = max_val(diff, window_size / 2 - 1);
        float diff_sum = 0.0f;
        for (int i = 0; i < window_size / 2 - 1; ++i) diff_sum += fabsf(diff[i]);

        features[base++] = fft_mean;
        features[base++] = fft_std;
        features[base++] = fft_max;
        features[base++] = fft_sum;
        features[base++] = diff_mean;
        features[base++] = diff_std;
        features[base++] = diff_max;
        features[base++] = diff_sum;
        features[base++] = entropy_hist(fft_mag, window_size / 2);
    }
}
void extract_top_features(float window[][6], int window_size, int8_t features[NUM_FEATURES]) {
    float raw_features[294];
    extract_features(window, window_size, raw_features);

    for (int i = 0; i < NUM_FEATURES; ++i) {
        int idx = TOP_FEATURE_INDICES[i];
        float standardized = (raw_features[idx] - FEATURE_MEAN[i]) / FEATURE_STD[i];
        int quantized = (int)roundf(standardized / INPUT_QUANT_SCALE + INPUT_QUANT_ZERO_POINT);

        // Clamp to int8 range
        if (quantized > 127) quantized = 127;
        else if (quantized < -128) quantized = -128;

        features[i] = (int8_t)quantized;
    }
}

