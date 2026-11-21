#include "preprocess.h"
#include "kws_config.h"
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include "esp_log.h"
#include "esp_dsp.h"

//Macro
#define PI 3.14159265358979323846f
#define KWS_SPECTRUM_BINS (KWS_N_FFT / 2 + 1)

static const char *TAG = "PREPROCESS";

//Hann Window: 400 samples
static float hann_window[KWS_FRAME_LEN];

//512 complex samples = 1024 floats
static float fft_buffer[KWS_N_FFT*2] __attribute__((aligned(16)));

//Power spectrum: 257 freq bins
static float power_spectrum[KWS_N_FFT/2+1];

// Sparse Mel filterbank: at most two non-zero triangular weights per FFT bin.
// Each Mel band stores a contiguous range of (FFT bin, weight) pairs.
#define KWS_MEL_MAX_WEIGHTS (2 * KWS_SPECTRUM_BINS)
static uint16_t mel_bin_indices[KWS_MEL_MAX_WEIGHTS];
static float mel_weights[KWS_MEL_MAX_WEIGHTS];
static uint16_t mel_offsets[KWS_N_MEL + 1];
static uint16_t mel_weight_count = 0;

//Mel energy: 40 bands
static float mel_energy[KWS_N_MEL];

//Log-Mel: 40 bands
static float log_mel[KWS_N_MEL];

//Log-Mel Spectrogram: 99 frames x 40 bands
static float log_mel_frames[KWS_N_FRAMES][KWS_N_MEL];


//Preprocess initialization state
static bool initialized = false;

/**
 * @brief Generate the periodic Hann window for each audio frame.
 *
 * Computes KWS_FRAME_LEN coefficients used to reduce spectral leakage
 * before applying the FFT. The window is generated once at initialization.
 */
static void generate_hann_window(void) {
    for (int n = 0; n < KWS_FRAME_LEN; n++) {
        float angle = 2.0f * PI * (float)n / (float)KWS_FRAME_LEN;
        hann_window[n] = 0.5f - 0.5f * cosf(angle);
    }
}

/**
 * @brief Calculate the power spectrum from the complex FFT output.
 *
 * Uses the squared magnitude (real^2 + imaginary^2) for each
 * nonnegative-frequency bin and stores the result in power_spectrum.
 */
static void compute_power_spectrum(void) {
    for(int k = 0; k < KWS_SPECTRUM_BINS; k++) {
        float real = fft_buffer[2*k];
        float imag = fft_buffer[2*k+1];

        power_spectrum[k] = real * real + imag*imag;
    }    
}

/**
 * @brief Convert a linear frequency in hertz to the Mel scale.
 *
 * @param hz Input frequency in hertz.
 * @return Corresponding frequency in Mel units.
 */
static float hz_to_mel(float hz) {
    return 1127.0f*logf(1.0f + hz/700.0f);
}

/**
 * @brief Build a sparse triangular Mel filterbank.
 *
 * The original 257 x 40 matrix contains mostly zero values. Store only
 * positive weights, grouped by Mel band and ordered by ascending FFT bin.
 * Each FFT bin overlaps at most two triangular filters. The numerical
 * weight formula is unchanged from the original dense implementation.
 *
 * @return ESP_OK on success, or ESP_ERR_INVALID_SIZE if capacity is exceeded.
 */
static esp_err_t generate_mel_filterbank(void) {
    const float mel_min = hz_to_mel(KWS_FMIN);
    const float mel_max = hz_to_mel(KWS_FMAX);
    const float mel_step = (mel_max - mel_min) / (KWS_N_MEL + 1);

    // Pass 1: count strictly positive weights in each Mel band.
    uint16_t counts[KWS_N_MEL] = {0};
    for (int k = 0; k < KWS_SPECTRUM_BINS; k++) {
        float freq = (float)k * KWS_SAMPLE_RATE / KWS_N_FFT;
        float mel_freq = hz_to_mel(freq);
        for (int m = 0; m < KWS_N_MEL; m++) {
            float left = mel_min + m * mel_step;
            float center = left + mel_step;
            float right = center + mel_step;
            float weight = 0.0f;
            if (k != 0) {
                float rising = (mel_freq - left) / (center - left);
                float falling = (right - mel_freq) / (right - center);
                weight = fmaxf(0.0f, fminf(rising, falling));
            }
            if (weight > 0.0f) {
                counts[m]++;
            }
        }
    }

    // Convert per-band counts into offsets (CSR-style representation).
    uint32_t total = 0;
    for (int m = 0; m < KWS_N_MEL; m++) {
        mel_offsets[m] = (uint16_t)total;
        total += counts[m];
    }
    if (total > KWS_MEL_MAX_WEIGHTS) {
        ESP_LOGE(TAG, "Mel filterbank capacity exceeded: %lu", (unsigned long)total);
        return ESP_ERR_INVALID_SIZE;
    }
    mel_offsets[KWS_N_MEL] = (uint16_t)total;
    mel_weight_count = (uint16_t)total;

    // Pass 2: fill (frequency bin, weight) pairs in ascending bin order.
    uint16_t positions[KWS_N_MEL];
    for (int m = 0; m < KWS_N_MEL; m++) {
        positions[m] = mel_offsets[m];
    }
    for (int k = 0; k < KWS_SPECTRUM_BINS; k++) {
        float freq = (float)k * KWS_SAMPLE_RATE / KWS_N_FFT;
        float mel_freq = hz_to_mel(freq);
        for (int m = 0; m < KWS_N_MEL; m++) {
            float left = mel_min + m * mel_step;
            float center = left + mel_step;
            float right = center + mel_step;
            float weight = 0.0f;
            if (k != 0) {
                float rising = (mel_freq - left) / (center - left);
                float falling = (right - mel_freq) / (right - center);
                weight = fmaxf(0.0f, fminf(rising, falling));
            }
            if (weight > 0.0f) {
                uint16_t pos = positions[m]++;
                mel_bin_indices[pos] = (uint16_t)k;
                mel_weights[pos] = weight;
            }
        }
    }
    return ESP_OK;
}

/**
 * @brief Accumulate energy only for non-zero weights in each Mel band.
 *
 * Sparse weights remain in ascending FFT-bin order so the summation order
 * of non-zero terms matches the original dense filterbank implementation.
 */
static void compute_mel_energy(void)
{
    for (int m = 0; m < KWS_N_MEL; m++) {
        float energy = 0.0f;
        for (uint16_t i = mel_offsets[m]; i < mel_offsets[m + 1]; i++) {
            energy += power_spectrum[mel_bin_indices[i]] * mel_weights[i];
        }
        mel_energy[m] = energy;
    }
}

/**
 * @brief Convert Mel-band energies into logarithmic features.
 *
 * Applies the natural logarithm to each Mel energy with a small
 * epsilon (1e-6) to avoid taking the logarithm of zero.
 */
static void compute_log_mel(void) {
    for(int m = 0; m < KWS_N_MEL; m++) {
        log_mel[m] = logf(mel_energy[m] + 1e-6f);
    }
}

/**
 * @brief Extract the Log-Mel features for one audio frame.
 *
 * Reads samples from either normalized float32 PCM or packed,
 * little-endian signed PCM16, then applies the Hann window,
 * zero-padding, FFT, power spectrum, Mel filters, and logarithm.
 * The 40 output values are saved in log_mel_frames[frame_index].
 *
 * @param pcm Float32 PCM array, or NULL when pcm16 is used.
 * @param pcm16 Raw PCM16 bytes, or NULL when pcm is used.
 * @param frame_index Frame index within the 2-second audio window.
 * @return ESP_OK on success; an error code if processing fails.
 */
static esp_err_t process_frame(const float *pcm, const uint8_t *pcm16, int frame_index) {
    int start = frame_index*KWS_HOP;

    //1. Hann Window + Zero padding
    for(int n = 0; n < KWS_N_FFT; n++) {
        float sample = 0.0f;
        if(n < KWS_FRAME_LEN) {
            float value = 0.0f;
            if(pcm != NULL) {
                value = pcm[start + n];
            }
            else {
                size_t index = (size_t)(start + n)*2;
                const uint8_t *p = pcm16 + index;
                uint16_t raw = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
                int32_t signed_sample = (raw < 0x8000u) ? (int32_t)raw : (int32_t)raw - 65536;
                //PCM to normalized float32
                value = (float)signed_sample/32768.0f;
            }
            sample = value*hann_window[n];
        }
        fft_buffer[2*n] = sample;
        fft_buffer[2*n+1] = 0.0f;
    }

    //2. FFT
    esp_err_t ret = dsps_fft2r_fc32(fft_buffer, KWS_N_FFT);
    if(ret != ESP_OK) return ret;

    //3. Bit reversal
    ret = dsps_bit_rev_fc32(fft_buffer, KWS_N_FFT);
    if(ret != ESP_OK) return ret;

    //4. Power Spectrum
    compute_power_spectrum();

    //5. Mel Energy
    compute_mel_energy();

    //6. Log-Mel
    compute_log_mel();

    //7. Save 40 Log-Mel values
    for (int m = 0; m < KWS_N_MEL; m++) {
        if (!isfinite(log_mel[m])) {
            return ESP_FAIL;
        }
        log_mel_frames[frame_index][m] = log_mel[m];
    }

    return ESP_OK;
}

/**
 * @brief Compute the global mean of the Log-Mel spectrogram.
 *
 * Averages all KWS_N_FRAMES * KWS_N_MEL feature values for
 * subsequent mean normalization.
 *
 * @return Mean Log-Mel value across the full audio window.
 */
static float compute_log_mel_mean(void) {
    double sum = 0.0f;
    for(int i = 0; i < KWS_N_FRAMES; i++) {
        for(int j = 0; j < KWS_N_MEL; j++) {
            sum += log_mel_frames[i][j];
        }
    }
    return sum/(KWS_N_FRAMES*KWS_N_MEL);
}

/**
 * @brief Initialize the audio preprocessing pipeline.
 *
 * Generates the Hann window, initializes FFT tables,
 * and builds the sparse Mel filterbank.
 * Initialization is idempotent.
 *
 * @return ESP_OK on success, or an ESP-IDF error code on failure.
 */
esp_err_t preprocess_init(void){
    if (initialized) {
        ESP_LOGI(TAG, "Preprocess already initialized");
        return ESP_OK;
    }

    ESP_LOGI(TAG, "===== PREPROCESS INITIALIZATION =====");
    generate_hann_window();
    ESP_LOGI(TAG, "Hann Window generated: %d samples", KWS_FRAME_LEN);

    esp_err_t ret = dsps_fft2r_init_fc32(NULL, KWS_N_FFT);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FFT initialization failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = generate_mel_filterbank();
    if (ret != ESP_OK) {
        return ret;
    }

    ESP_LOGI(TAG, "Sparse Mel Filterbank: %u non-zero weights", (unsigned)mel_weight_count);
    initialized = true;
    ESP_LOGI(TAG, "===== PREPROCESS READY =====");
    return ESP_OK;
}

/**
 * @brief Run the shared audio-to-INT8 feature extraction pipeline.
 *
 * Processes all frames, computes the global Log-Mel mean, applies
 * 
 *  * mean normalization and feature scaling, then quantizes to INT8
 * with the model input scale and zero point.
 * Exactly one of pcm and pcm16 must be non-NULL.
 *
 * @param pcm Normalized float32 PCM samples, or NULL.
 * @param pcm16 Packed little-endian PCM16 bytes, or NULL.
 * @param sample_count Number of audio samples (not bytes).
 * @param output_features Destination INT8 feature array.
 * @param output_count Number of available INT8 output elements.
 * @return ESP_OK on success; an error code for invalid input or failure.
 */
static esp_err_t preprocess_extract_internal(const float *pcm, const uint8_t *pcm16, size_t sample_count, int8_t *output_features, size_t output_count) {
    if (!initialized) {
        return ESP_ERR_INVALID_STATE;
    }
    if (output_features == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    //Exactly one audio input must be provided
    if((pcm == NULL) == (pcm16 == NULL)) {
        return ESP_ERR_INVALID_ARG;
    }

    if (sample_count != KWS_WINDOW_SAMPLES || output_count != KWS_N_FRAMES * KWS_N_MEL) {
        return ESP_ERR_INVALID_SIZE;
    }

    // Process 99 frames
    for (int frame = 0; frame < KWS_N_FRAMES; frame++) {
        esp_err_t ret = process_frame(pcm, pcm16, frame);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to process frame %d", frame);
            return ret;
        }
    }

    //6. Mean Normalization
    float mean = compute_log_mel_mean();

    if (!isfinite(mean)) {
        ESP_LOGE(TAG, "Invalid Log-Mel mean");
        return ESP_FAIL;
    }

    ESP_LOGD(TAG, "Log-Mel mean = %.6f", (double)mean);
    ESP_LOGD(TAG, "Processed %d Log-Mel frames", KWS_N_FRAMES);
    ESP_LOGD(TAG, "Log-Mel[0][0] = %.6f", (double)log_mel_frames[0][0]);
    ESP_LOGD(TAG, "Log-Mel[98][39] = %.6f", (double)log_mel_frames[98][39]);

    //7. Mean Normalization + INT8 Quantization
    for (int t = 0; t < KWS_N_FRAMES; t++) {
        for (int m = 0; m < KWS_N_MEL; m++) {

            // Mean normalization + Feature scaling
            float normalized = (log_mel_frames[t][m] - mean) * KWS_FEATURE_SCALE;
            // Check normalized feature
            if (!isfinite(normalized)) {
                ESP_LOGE(TAG,"Invalid feature [%d][%d]", t, m);
                return ESP_FAIL;
            }

            // INT8 quantization
            float q = nearbyintf(normalized / KWS_IN_SCALE + KWS_IN_ZERO_POINT);

            // Clip to INT8 range
            q = fmaxf(-128.0f, fminf(127.0f, q));

            // Save output features
            output_features[t * KWS_N_MEL + m] = (int8_t)q;
        }
    }

    return ESP_OK;
}

/*Existing API: PCM float32*/
/**
 * @brief Extract quantized Log-Mel features from float32 PCM audio.
 *
 * Retains the original float32 entry point used for reference tests.
 *
 * @param pcm Normalized float32 PCM array.
 * @param sample_count Number of samples in pcm.
 * @param output_features Destination INT8 features.
 * @param output_count Number of INT8 elements available.
 * @return ESP_OK on success; an error code otherwise.
 */
esp_err_t preprocess_extract(const float *pcm, size_t sample_count, int8_t *output_features, size_t output_count) {
    return preprocess_extract_internal(pcm, NULL, sample_count, output_features, output_count);
}

/*PCM16 input API*/
/**
 * @brief Extract quantized Log-Mel features directly from PCM16 bytes.
 *
 * Decodes signed little-endian PCM16 samples as needed for each frame,
 * avoiding a separate full-size float32 audio buffer in RAM.
 *
 * @param pcm_bytes Pointer to raw PCM16 data (without the WAV header).
 * @param sample_count Number of PCM16 samples, not the byte count.
 * @param output_features Destination INT8 features.
 * @param output_count Number of INT8 elements available.
 * @return ESP_OK on success; an error code otherwise.
 */
esp_err_t preprocess_extract_pcm16(const uint8_t *pcm_bytes, size_t sample_count, int8_t *output_features, size_t output_count) {
    return preprocess_extract_internal(NULL, pcm_bytes, sample_count, output_features, output_count);
}