
#include <stdio.h>
#include "esp_log.h"
#include "esp_err.h"
#include <stdlib.h>
#include <math.h>
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

//#include "fan.h"
#include "display.h"
#include "command.h"
#include "preprocess.h"
#include "kws_config.h"
#include "wav_reader.h"
#include "audio.h"
#include "encoder.h"
//#include "pid_controller.h"

#define MIC_READ_SAMPLES 256
static const char *TAG = "MIC_KWS_TEST";
static int32_t i2s_chunk[MIC_READ_SAMPLES];
static int8_t features[KWS_N_FRAMES * KWS_N_MEL];

// 24-bit signed microphone values in the high bits of an I2S 32-bit slot:
// retain the upper 16 bits as signed PCM16. Check the mic datasheet if different.
static int16_t i2s_to_pcm16(int32_t value)
{
      
    return (int16_t)(value >> 16);
}

void app_main(void)
{
    display_draw_char(8, 24, 'A', true);
    ESP_LOGI(TAG, "===== MICROPHONE -> PREPROCESS -> KWS =====");
    // Initialize encoder
    ESP_ERROR_CHECK(encoder_init());
    ESP_ERROR_CHECK(encoder_reset());

    ESP_LOGI("ENCODER_TEST", "Encoder initialized");
    esp_err_t ret = preprocess_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "preprocess_init: %s", esp_err_to_name(ret));
        return;
    }
    ret = command_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "command_init: %s", esp_err_to_name(ret));
        return;
    }
    // Preprocess takes little-endian PCM16 bytes; reserve exactly 2 seconds.
    const size_t pcm_bytes = (size_t)KWS_WINDOW_SAMPLES * sizeof(int16_t);
    uint8_t *pcm = heap_caps_malloc(pcm_bytes, MALLOC_CAP_8BIT);
    if (!pcm) {
        ESP_LOGE(TAG, "Not enough RAM for %u-byte audio buffer", (unsigned)pcm_bytes);
        return;
    }

    ret = audio_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "audio_init: %s", esp_err_to_name(ret));
        heap_caps_free(pcm);
        return;
    }

    unsigned trial = 0;
    while (true) {
        ++trial;
        int encoder_count = 0;

        esp_err_t enc_ret = encoder_get_count(&encoder_count);

        if (enc_ret == ESP_OK) {
            ESP_LOGI("ENCODER_TEST",
                    "Encoder count = %d",
                    encoder_count);
        } 
        else {
            ESP_LOGE("ENCODER_TEST",
                    "Failed to read encoder");
        }
        size_t collected = 0;
        int32_t min_pcm = INT16_MAX;
        int32_t max_pcm = INT16_MIN;
        uint64_t sum_abs = 0;
        int nonzero = 0;
        int clipped = 0;
        int64_t sum_sq = 0;
        ESP_LOGI(TAG, "Capture #%u: speak a command now", trial);

        while (collected < KWS_WINDOW_SAMPLES) {
            size_t requested = KWS_WINDOW_SAMPLES - collected;
            if (requested > MIC_READ_SAMPLES) requested = MIC_READ_SAMPLES;
            size_t bytes_read = 0;
            ret = audio_read(i2s_chunk, requested * sizeof(int32_t), &bytes_read);
            if (ret != ESP_OK || bytes_read == 0 || bytes_read % sizeof(int32_t) != 0) {
                ESP_LOGE(TAG, "audio_read: %s, bytes=%u",
                         esp_err_to_name(ret), (unsigned)bytes_read);
                goto stop;
            }
            size_t n = bytes_read / sizeof(int32_t);
            // In 10 mẫu I2S khác 0 đầu tiên
            static int raw_printed = 0;

            for (size_t i = 0; i < n && raw_printed < 10; i++) {
                if (i2s_chunk[i] != 0) {
                    ESP_LOGI("I2S_DEBUG",
                            "RAW = %ld (0x%08lX), PCM16 = %d",
                            (long)i2s_chunk[i],
                            (unsigned long)(uint32_t)i2s_chunk[i],
                            (int)i2s_to_pcm16(i2s_chunk[i]));

                    raw_printed++;
                }
            }
            static bool printed = false;

            if (!printed) {
                printed = true;

                for (size_t i = 0; i < n && i < 10; i++) {
                    ESP_LOGI("I2S_RAW",
                            "RAW[%u] = %ld (0x%08lX)",
                            (unsigned)i,
                            (long)i2s_chunk[i],
                            (unsigned long)(uint32_t)i2s_chunk[i]);
                }
            }
            for (size_t i = 0; i < n; ++i) {
                int16_t sample = i2s_to_pcm16(i2s_chunk[i]);
                if (sample != 0) {
                    nonzero++;
                }

                if (sample == 32767 || sample == -32768) {
                    clipped++;
                }

                sum_sq += (int64_t)sample * sample;
                if (sample < min_pcm) min_pcm = sample;
                if (sample > max_pcm) max_pcm = sample;
                int32_t s = (int32_t)sample;
                sum_abs += (uint32_t)(s < 0 ? -s : s);
                size_t offset = (collected + i) * 2;
                pcm[offset] = (uint8_t)((uint16_t)sample & 0xFF);
                pcm[offset + 1] = (uint8_t)(((uint16_t)sample >> 8) & 0xFF);
            }
            collected += n;
        }

        ESP_LOGI(TAG, "Audio samples=%u, min=%ld, max=%ld, mean_abs=%lu",
                 (unsigned)collected, (long)min_pcm, (long)max_pcm,
                 (unsigned long)(sum_abs / collected));

        float rms = sqrtf((float)sum_sq / collected);

        ESP_LOGI("AUDIO_CHECK",
                "nonzero=%d, clipped=%d, RMS=%.2f",
                nonzero, clipped, (double)rms);
        ret = preprocess_extract_pcm16(pcm, collected, features, sizeof(features));
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "preprocess: %s", esp_err_to_name(ret));
            goto stop;
        }

        int predicted_class = -1;
        float confidence = 0;
        ret = command_predict(features, sizeof(features), &predicted_class, &confidence);
        if (confidence < 0.80f) {
            ESP_LOGW(TAG,
                    "Prediction rejected: confidence=%.4f",
                    (double)confidence);
        } 
        else {
            ESP_LOGI(TAG,
                    "Prediction accepted: %s",
                    KWS_LABELS[predicted_class]);
        }
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "command_predict: %s", esp_err_to_name(ret));
            goto stop;
        }
                if (predicted_class >= 0 && predicted_class < KWS_NUM_CLASSES) {
            ESP_LOGI(TAG, "Prediction #%u: class=%d, label=%s, confidence=%.4f",
                     trial, predicted_class, KWS_LABELS[predicted_class], (double)confidence);
        } else {
            ESP_LOGW(TAG, "Prediction #%u: invalid class=%d", trial, predicted_class);
        }
        // Allows a moment to read the result; inference is not simultaneous with capture.
        vTaskDelay(pdMS_TO_TICKS(5000));
    }

stop:
    heap_caps_free(pcm);
    ESP_LOGE(TAG, "Microphone KWS test stopped");
}