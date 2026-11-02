#include "wav_reader.h"
#include "esp_log.h"
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#define WAV_RIFF_HEADER_BYTES 12u
#define WAV_CHUNK_HEADER_BYTES 8u
#define WAV_PCM_FORMAT 1u
#define WAV_REQUIRED_CHANNELS 1u
#define WAV_REQUIRED_RATE 16000u
#define WAV_REQUIRED_BITS 16u
#define WAV_PCM16_BYTES_PER_SAMPLE 2u

static const char *TAG = "WAV_READER";

/**
 * @brief Read an unsigned 16-bit little-endian integer.
 *
 * Reading individual bytes avoids unaligned accesses and does not depend
 * on the CPU's native byte order.
 *
 * @param p Pointer to at least two valid bytes.
 * @return Decoded 16-bit unsigned value.
 */
static uint16_t read_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

/**
 * @brief Read an unsigned 32-bit little-endian integer.
 *
 * @param p Pointer to at least four valid bytes.
 * @return Decoded 32-bit unsigned value.
 */
static uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/**
 * @brief Parse a RIFF/WAVE buffer and expose its PCM16 payload without copying.
 *
 * Validates header and chunk boundaries before accessing their fields.
 * Supports additional RIFF chunks and word-aligned chunk padding.
 * The output points into the original WAV buffer, typically embedded in flash.
 *
 * @param wav_data Complete WAV file data.
 * @param wav_size Available bytes in the WAV buffer.
 * @param reader Descriptor filled with PCM pointer, size, rate and count.
 * @return ESP_OK, ESP_ERR_INVALID_ARG, ESP_ERR_INVALID_SIZE,
 *         ESP_ERR_INVALID_RESPONSE, or ESP_ERR_NOT_SUPPORTED.
 */
esp_err_t wav_reader_open( const uint8_t *wav_data, size_t wav_size, wav_reader_t *reader) {
    if (!wav_data || !reader) {
        return ESP_ERR_INVALID_ARG;
    }

    memset(reader, 0, sizeof(*reader));

    if (wav_size < WAV_RIFF_HEADER_BYTES) {
        return ESP_ERR_INVALID_SIZE;
    }

    // Check RIFF/WAVE header
    if (memcmp(wav_data, "RIFF", 4) != 0 || memcmp(wav_data + 8, "WAVE", 4) != 0) {
        ESP_LOGE(TAG, "Invalid RIFF/WAVE header");
        return ESP_ERR_INVALID_RESPONSE;
    }

    // RIFF size excludes the first eight bytes of the file
    uint32_t riff_size = read_u32_le(wav_data + 4);
    if (riff_size < 4 || (size_t)riff_size > wav_size - 8) {
        return ESP_ERR_INVALID_SIZE;
    }

    size_t end = (size_t)riff_size + 8;
    size_t pos = WAV_RIFF_HEADER_BYTES;

    bool fmt_found = false;
    bool data_found = false;

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint16_t bits_per_sample = 0;
    uint16_t block_align = 0;
    uint32_t sample_rate = 0;

    const uint8_t *pcm_data = NULL;
    size_t pcm_bytes = 0;

    // Search WAV chunks
    while (pos <= end && end - pos >= WAV_CHUNK_HEADER_BYTES) {
        const uint8_t *chunk = wav_data + pos;
        uint32_t chunk_size = read_u32_le(chunk + 4);

        pos += WAV_CHUNK_HEADER_BYTES;

        if ((size_t)chunk_size > end - pos) {
            return ESP_ERR_INVALID_SIZE;
        }

        if (memcmp(chunk, "fmt ", 4) == 0) {

            if (chunk_size < 16) {
                return ESP_ERR_INVALID_SIZE;
            }

            const uint8_t *fmt = wav_data + pos;
            audio_format = read_u16_le(fmt);
            channels = read_u16_le(fmt + 2);
            sample_rate = read_u32_le(fmt + 4);
            block_align = read_u16_le(fmt + 12);
            bits_per_sample = read_u16_le(fmt + 14);
            fmt_found = true;
        }

        if (memcmp(chunk, "data", 4) == 0 && !data_found) {
            // Zero-copy: keep a pointer to the existing PCM payload
            pcm_data = wav_data + pos;
            pcm_bytes = chunk_size;
            data_found = true;
        }

        // RIFF chunks have a padding byte when their payload size is odd
        size_t padded_size = (size_t)chunk_size + (chunk_size & 1u);

        if (padded_size > end - pos) {
            return ESP_ERR_INVALID_SIZE;
        }

        pos += padded_size;
    }

    if (!fmt_found || !data_found) {
        ESP_LOGE(TAG, "Missing fmt or data chunk");
        return ESP_ERR_INVALID_RESPONSE;
    }

    ESP_LOGI(TAG, "Audio format: %u", audio_format);
    ESP_LOGI(TAG, "Channels: %u", channels);
    ESP_LOGI(TAG, "Sample rate: %lu", (unsigned long)sample_rate);
    ESP_LOGI(TAG, "Bits per sample: %u", bits_per_sample);

    // Only support PCM16 mono 16 kHz
    if (audio_format != WAV_PCM_FORMAT || 
        channels != WAV_REQUIRED_CHANNELS  || 
        sample_rate != WAV_REQUIRED_RATE  || 
        bits_per_sample != WAV_REQUIRED_BITS  || 
        block_align != WAV_PCM16_BYTES_PER_SAMPLE
    ) {
        ESP_LOGE(TAG, "Unsupported WAV format");
        return ESP_ERR_NOT_SUPPORTED;
    }

    if (pcm_bytes == 0 || pcm_bytes % WAV_PCM16_BYTES_PER_SAMPLE  != 0) {
        return ESP_ERR_INVALID_SIZE;
    }

    reader->pcm_data = pcm_data;
    reader->pcm_bytes = pcm_bytes;
    reader->sample_rate = sample_rate;
    reader->sample_count = pcm_bytes / WAV_PCM16_BYTES_PER_SAMPLE;

    ESP_LOGI(TAG, "PCM bytes: %u", (unsigned)pcm_bytes);
    ESP_LOGI(TAG, "Total samples: %lu", (unsigned long)reader->sample_count);
    ESP_LOGI(TAG, "WAV initialized successfully");

    return ESP_OK;
}

/**
 * @brief Read one signed PCM16 sample and normalize it to float32.
 *
 * @param reader Initialized reader containing the PCM payload pointer.
 * @param index Sample index, in the range [0, sample_count).
 * @param sample Output for the normalized sample value.
 * @return ESP_OK on success, or an error code if arguments are invalid.
 */
esp_err_t wav_reader_get_sample( const wav_reader_t *reader, size_t index, float *sample)
{
    if (!reader || !sample || !reader->pcm_data) {
        return ESP_ERR_INVALID_ARG;
    }
    if (index >= reader->sample_count) {
        return ESP_ERR_INVALID_SIZE;
    }

    // Each PCM16 sample occupies two bytes
    const uint8_t *p = reader->pcm_data + index * WAV_PCM16_BYTES_PER_SAMPLE;

    uint16_t raw = read_u16_le(p);

    // Sign extension to 32-bit
    int32_t signed_sample = (raw < 0x8000u) ? (int32_t)raw : (int32_t)raw - 65536;

    // Normalize PCM16 to float32
    *sample = (float)signed_sample / 32768.0f;

    return ESP_OK;
}