#ifndef WAV_READER_H
#define WAV_READER_H

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

/**
 * @brief View of PCM16 audio stored inside a WAV buffer.
 *
 * The reader does not own or copy the audio data. The original WAV buffer
 * must remain valid for as long as pcm_data is used.
 */
typedef struct {
    const uint8_t *pcm_data;
    size_t pcm_bytes;
    uint32_t sample_rate;
    uint32_t sample_count;
} wav_reader_t;

/**
 * @brief Validate a WAV buffer and locate its PCM16 audio payload.
 *
 * Parses the RIFF/WAVE header and searches for the "fmt " and "data" chunks,
 * including any intervening chunks. Only uncompressed, mono, 16-bit PCM
 * at 16 kHz is supported.
 *
 * The function is zero-copy: reader->pcm_data points directly into wav_data.
 * No heap allocation or PCM conversion is performed.
 *
 * @param wav_data Pointer to the complete WAV file in memory or flash.
 * @param wav_size Total available buffer size in bytes.
 * @param reader Output descriptor; cleared before parsing when valid.
 * @return ESP_OK on success, or an ESP-IDF error code on failure.
 */
esp_err_t wav_reader_open(const uint8_t *wav_data, size_t wav_size, wav_reader_t *reader);

/**
 * @brief Decode a single PCM16 sample as a normalized float32 value.
 *
 * Converts a signed little-endian 16-bit sample to the range [-1.0, 1.0).
 * This helper is optional: preprocess_extract_pcm16() can read pcm_data
 * directly without allocating or converting an entire float32 buffer.
 *
 * @param reader Initialized WAV reader descriptor.
 * @param index Zero-based sample index.
 * @param sample Output pointer receiving the normalized sample.
 * @return ESP_OK on success, or an ESP-IDF error code on failure.
 */
esp_err_t wav_reader_get_sample(const wav_reader_t *reader, size_t index, float *sample);

#endif