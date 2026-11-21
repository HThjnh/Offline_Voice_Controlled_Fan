#pragma once
// Tham số đặc trưng - phải khớp tuyệt đối với train_kws.py
#define KWS_SAMPLE_RATE 16000
#define KWS_WINDOW_SAMPLES 32000
#define KWS_N_FFT 512
#define KWS_FRAME_LEN 400
#define KWS_HOP 320
#define KWS_N_MEL 40
#define KWS_N_FRAMES 99
#define KWS_FMIN 60.0f
#define KWS_FMAX 7600.0f
#define KWS_FEATURE_SCALE 0.25f
#define KWS_IN_SCALE 0.024742744863033295f
#define KWS_IN_ZERO_POINT -30
#define KWS_OUT_SCALE 0.13761857151985168f
#define KWS_OUT_ZERO_POINT -14
#define KWS_NUM_CLASSES 8
static const char* const KWS_LABELS[KWS_NUM_CLASSES] = {"background", "unknown", "turn_on", "turn_off", "low", "medium", "high", "max"};
