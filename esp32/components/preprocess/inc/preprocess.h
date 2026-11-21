#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the audio preprocessing pipeline.
 *
 * Generates the Hann window, initializes the FFT,
 * and builds the sparse Mel filterbank.
 * Must be called before either feature-extraction function.
 *
 * @return ESP_OK on success, or an ESP-IDF error code on failure.
 */
esp_err_t preprocess_init(void);

/**
 * @brief Extract quantized Log-Mel features from float32 PCM audio.
 *
 * Processes a 2-second, 16-kHz mono audio window using overlapping frames,
 * Hann windowing, FFT, Mel filtering, log compression, mean normalization,
 * and INT8 quantization. Audio samples are expected to be normalized floats.
 *
 * @param pcm Pointer to the float32 PCM sample buffer.
 * @param sample_count Number of audio samples (KWS_WINDOW_SAMPLES).
 * @param output_features Destination buffer for quantized INT8 features.
 * @param output_count Number of output elements; must equal KWS_N_FRAMES * KWS_N_MEL.
 * @return ESP_OK on success, or an ESP-IDF error code on failure.
 */
esp_err_t preprocess_extract(
    const float *pcm,
    size_t sample_count,
    int8_t *output_features,
    size_t output_count
);

/**
 * @brief Extract quantized Log-Mel features directly from PCM16 data.
 *
 * Reads signed, little-endian 16-bit PCM samples and normalizes them as
 * needed for each frame, avoiding a separate full-length float32 buffer.
 * The output follows the same preprocessing and quantization steps as
 * preprocess_extract(). Pass raw PCM bytes, not the WAV file header.
 *
 * @param pcm_bytes Pointer to the raw little-endian PCM16 audio bytes.
 * @param sample_count Number of PCM16 samples, not the number of bytes.
 * @param output_features Destination buffer for quantized INT8 features.
 * @param output_count Number of output elements; must equal KWS_N_FRAMES * KWS_N_MEL.
 * @return ESP_OK on success, or an ESP-IDF error code on failure.
 */
esp_err_t preprocess_extract_pcm16(
    const uint8_t *pcm_bytes,
    size_t sample_count,
    int8_t *output_features,
    size_t output_count
);

#ifdef __cplusplus
}
#endif

#endif