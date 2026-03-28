//WORKING
#ifndef FEATURES_H
#define FEATURES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif



#define NUM_FEATURES 30
#define INPUT_QUANT_SCALE 0.0338241308927536f
#define INPUT_QUANT_ZERO_POINT -45.0f


// Top 30 features selected by RandomForest + MI
static const int TOP_FEATURE_INDICES[NUM_FEATURES] = {
    92, 55, 89, 49, 75, 76, 54, 97, 90, 71,
    82, 72, 83, 60, 95, 74, 52, 69, 61, 70,
    35, 40, 96, 165, 45, 30, 161, 41, 53, 77
};

// Mean of each selected feature (from StandardScaler)
static const float FEATURE_MEAN[NUM_FEATURES] = {
    4354.67886f, -214.58637f, 49455.12888f, -181.92314f, 4532.56117f,
    49474.11269f, -176.14763f, 49474.11269f, -387.55186f, 50474.85692f,
    25747073.14041f, 13.41445f, 24858089.62716f, 212.47342f, 73276.67243f,
    734.05190f, -341.01837f, 3399.57476f, 14203842.51133f, 359.28781f,
    7.51305f, 13002.87124f, 69284.41873f, 12025.37007f, 15615.26523f,
    -227.44607f, 12025.37007f, -92.44029f, -68.72269f, 93958.64358f
};

// Standard deviation of each selected feature (from StandardScaler)
static const float FEATURE_STD[NUM_FEATURES] = {
    2084.23087f, 132.41510f, 22643.45575f, 109.52206f, 2077.01201f,
    22646.79538f, 108.08559f, 22646.79538f, 180.62942f, 22738.34069f,
    15777730.08252f, 4.24198f, 14887627.63121f, 101.67991f, 51755.37310f,
    591.73585f, 312.87126f, 1626.87860f, 10229817.90713f, 320.74847f,
    3.36002f, 18572.42194f, 71435.48296f, 22964.93111f, 18989.52463f,
    226.00036f, 22964.93111f, 149.45141f, 221.95777f, 75742.18889f
};

// Extract full 294 features
void extract_features(float window[][6], int window_size, float features[294]);




void extract_top_features(float window[][6], int window_size, int8_t features[NUM_FEATURES]);

#ifdef __cplusplus
}
#endif

#endif  // FEATURES_H



////////////////////////////
//////////////////////////
/////////////////////////////
////////////////////////////
/////////////////////////
/////////////////////////

