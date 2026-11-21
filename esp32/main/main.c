#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "audio.h"
#include "command.h"
#include "control.h"
#include "kws_config.h"
#include "preprocess.h"

// Continuous capture configuration.
#define MIC_READ_SAMPLES             256
#define AUDIO_TASK_STACK_BYTES       4096
#define AUDIO_TASK_PRIORITY          7

// Enable motor control when set to 1; use 0 for KWS-only testing.
#define KWS_MOTOR_ENABLED            1
#define KWS_OLED_ENABLED             1
#define KWS_CONFIDENCE_MIN           0.80f
#define KWS_COMMAND_COOLDOWN_MS      1500
#define OLED_REFRESH_MS              1500
#define OLED_RESULT_HOLD_MS          1800

// The FIFO holds 1 second; the PCM window holds the original 2 seconds.
// The FIFO is a bounded queue, not the full KWS window.
#define AUDIO_FIFO_SAMPLES           KWS_SAMPLE_RATE
#define KWS_SLIDE_SAMPLES            (KWS_SAMPLE_RATE / 4)

// The current preprocessing configuration must fit inside its PCM window.
#define KWS_ACTIVE_SAMPLES           (((KWS_N_FRAMES) - 1) * (KWS_HOP) + (KWS_FRAME_LEN))

#if KWS_OLED_ENABLED
#include "display.h"
#endif

static const char *TAG = "MAIN";

// Shared FIFO: the audio task writes; app_main consumes.
static int16_t *audio_fifo = NULL;
static int16_t *pcm_window = NULL;
static int8_t features[KWS_N_FRAMES * KWS_N_MEL];

static SemaphoreHandle_t fifo_mutex = NULL;
static SemaphoreHandle_t capture_done = NULL;
static TaskHandle_t kws_task_handle = NULL;

static size_t fifo_read_pos = 0;
static size_t fifo_write_pos = 0;
static size_t fifo_count = 0;
static uint32_t fifo_overrun_events = 0;
static uint32_t fifo_dropped_samples = 0;

static volatile bool capture_enabled = false;
static volatile esp_err_t capture_error = ESP_OK;

#if KWS_OLED_ENABLED
// Keep OLED updates in app_main (one writer); the audio and PI tasks
// must never wait on I2C transactions.
static bool oled_initialized = false;
// Store the last successfully accepted voice command.
static char oled_last_command[12] = "NONE";
static TickType_t oled_last_draw_tick = 0;
static TickType_t oled_result_start_tick = 0;
static bool oled_result_visible = false;

/**
 * Update the OLED with the current state, last accepted command, and fan level.
 *
 * @param state       Short status text shown on the second display line.
 * @param confidence  KWS confidence; used only in recognition-only mode.
 * @param immediate   Bypass the normal OLED refresh interval when true.
 * @param hold_result Keep the displayed result visible for a short period.
 */
static void oled_draw_status(const char *state, const char *label, float confidence, bool immediate, bool hold_result) {
    if (!oled_initialized) {
        return;
    }
    
    const TickType_t now = xTaskGetTickCount();
    if(!immediate && (TickType_t)(now - oled_last_draw_tick) < pdMS_TO_TICKS(OLED_REFRESH_MS)) {
        return;
    }

    char command_line[17];
    char detail_line[17];
    // The latest KWS label may be background or unknown.
    // Display the last accepted command instead.
    (void)label;
    snprintf(command_line, sizeof(command_line), "STATUS: %.8s", oled_last_command);

    // Convert lowercase letters to uppercase.
    // Replace underscores with spaces.
    for (size_t i = 5; command_line[i] != '\0'; i++) {
        if (command_line[i] >= 'a' && command_line[i] <= 'z') {
            command_line[i] = command_line[i] - 'a' + 'A';
        }
        if (command_line[i] == '_') {
            command_line[i] = ' ';
        }
    }

#if KWS_MOTOR_ENABLED
    (void)confidence;
    // Read the current target RPM from the PI controller.
    const float target_rpm = control_get_target_rpm();
    
    // Show fixed, user-friendly speed levels.
    if (target_rpm <= 0.0f) {
        strcpy(detail_line, "POWER: 0%");
    }
    else if (target_rpm <= 80.0f) {
        strcpy(detail_line, "POWER: 40%");
    }
    else if (target_rpm <= 110.0f) {
        strcpy(detail_line, "POWER: 55%");
    }
    else if (target_rpm <= 140.0f) {
        strcpy(detail_line, "POWER: 70%");
    }
    else if (target_rpm <= 170.0f) {
        strcpy(detail_line, "POWER: 85%");
    }
    else {
        strcpy(detail_line, "POWER: 100%");
    }

#else
    // Display KWS confidence in recognition-only mode
    if (confidence >= 0.0f) {
        unsigned pct = (unsigned)(confidence * 100.0f + 0.5f);
        if (pct > 100U) {
            pct = 100U;
        }
        snprintf(detail_line, sizeof(detail_line), "CONF: %u%%", pct);
    } 
    else {
        snprintf(detail_line, sizeof(detail_line), "MOTOR: TEST");
    }
#endif

    // display_clear() also sends a blank framebuffer in the existing driver.
    // Keep refreshes sparse so OLED I2C traffic does not slow down KWS.
    display_clear();

    display_draw_string(0, "VOICE FAN", true);
    display_draw_string(16, state, true);
    display_draw_string(32, command_line, true);
    display_draw_string(48, detail_line, true);
    display_update();

    oled_last_draw_tick = xTaskGetTickCount();
    if (hold_result) {
        oled_result_visible = true;
        oled_result_start_tick = oled_last_draw_tick;
    }
}

/**
 * Show the listening state unless a recently accepted result is still visible.
 */
static void oled_show_listening(const char *label, float confidence) {
    if (oled_result_visible) {
        if ((TickType_t)(xTaskGetTickCount() - oled_result_start_tick) < pdMS_TO_TICKS(OLED_RESULT_HOLD_MS)) {
            return;
        }
        oled_result_visible = false;
    }
    oled_draw_status("LISTENING", label, confidence, false, false);
}

/**
 * Show a low-confidence rejection, subject to the result hold and refresh limits.
 */
static void oled_show_rejected(const char *label, float confidence) {
    if (oled_result_visible && (TickType_t)(xTaskGetTickCount() - oled_result_start_tick) < pdMS_TO_TICKS(OLED_RESULT_HOLD_MS)) {
        return;
    }

    // Low-confidence actionable label: only show occasionally.
    if ((TickType_t)(xTaskGetTickCount() - oled_last_draw_tick) >= pdMS_TO_TICKS(OLED_REFRESH_MS)) {
        oled_draw_status("REJECTED", label, confidence, true, true);
    }
}
#endif

/**
 * Convert a signed 32-bit I2S sample to 16-bit PCM using its upper 16 bits.
 */
static inline int16_t i2s_to_pcm16(int32_t value) {
    return (int16_t)(value >> 16);
}

/**
 * Convert and append an I2S chunk to the shared PCM FIFO.
 *
 * If the FIFO lacks space, discard the oldest samples and record an overrun
 * so app_main can reset its audio window instead of mixing discontinuous data.
 *
 * @param input   Input buffer containing signed 32-bit I2S samples.
 * @param samples Number of input samples to append.
 */
static void fifo_push_i2s(const int32_t *input, size_t samples)
{
    xSemaphoreTake(fifo_mutex, portMAX_DELAY);

    if (fifo_count + samples > AUDIO_FIFO_SAMPLES) {
        const size_t drop = fifo_count + samples - AUDIO_FIFO_SAMPLES;
        fifo_read_pos = (fifo_read_pos + drop) % AUDIO_FIFO_SAMPLES;
        fifo_count -= drop;
        fifo_dropped_samples += (uint32_t)drop;
        ++fifo_overrun_events;
    }

    for (size_t i = 0; i < samples; ++i) {
        audio_fifo[fifo_write_pos] = i2s_to_pcm16(input[i]);
        fifo_write_pos = (fifo_write_pos + 1) % AUDIO_FIFO_SAMPLES;
    }
    fifo_count += samples;

    xSemaphoreGive(fifo_mutex);

    // Notify on every chunk; notifications coalesce while inference runs.
    // Do not print from this high-priority task for every chunk.
    xTaskNotifyGive(kws_task_handle);
}

/**
 * Continuously read microphone chunks and push converted PCM samples into the FIFO.
 *
 * The task notifies app_main when new samples arrive or when microphone capture fails.
 */
static void audio_capture_task(void *arg) {
    (void)arg;
    int32_t chunk[MIC_READ_SAMPLES];

    while (capture_enabled) {
        size_t bytes_read = 0;
        esp_err_t ret = audio_read(chunk, sizeof(chunk), &bytes_read);

        if (ret != ESP_OK || bytes_read == 0 || bytes_read % sizeof(int32_t) != 0 || bytes_read > sizeof(chunk)) {
            capture_error = (ret == ESP_OK) ? ESP_FAIL : ret;
            ESP_LOGE(TAG, "Microphone read failed: %s", esp_err_to_name(capture_error));
            xTaskNotifyGive(kws_task_handle);
            break;
        }
        fifo_push_i2s(chunk, bytes_read / sizeof(int32_t));
    }

    xSemaphoreGive(capture_done);
    vTaskDelete(NULL);
}

/**
 * Read the queued sample count and FIFO overrun statistics under the mutex.
 *
 * @param overrun_generation Receives the current overrun counter.
 * @param dropped_samples    Receives the cumulative number of discarded samples.
 * @return Number of samples currently available in the FIFO.
 */
static size_t fifo_available(uint32_t *overrun_generation, uint32_t *dropped_samples) {
    xSemaphoreTake(fifo_mutex, portMAX_DELAY);
    const size_t available = fifo_count;
    *overrun_generation = fifo_overrun_events;
    *dropped_samples = fifo_dropped_samples;
    xSemaphoreGive(fifo_mutex);
    return available;
}

/**
 * Copy up to the requested number of oldest samples from the FIFO.
 *
 * @param dst                Destination PCM buffer.
 * @param wanted             Maximum number of samples to copy.
 * @param overrun_generation Receives the FIFO overrun counter observed during the copy.
 * @return Number of samples copied.
 */
static size_t fifo_pop(int16_t *dst, size_t wanted, uint32_t *overrun_generation) {
    xSemaphoreTake(fifo_mutex, portMAX_DELAY);

    size_t taken = (wanted < fifo_count) ? wanted : fifo_count;
    const size_t first = (taken < AUDIO_FIFO_SAMPLES - fifo_read_pos) ? taken : AUDIO_FIFO_SAMPLES - fifo_read_pos;

    if (first > 0) {
        memcpy(dst, audio_fifo + fifo_read_pos, first * sizeof(int16_t));
    }
    if (taken > first) {
        memcpy(dst + first, audio_fifo, (taken - first) * sizeof(int16_t));
    }

    fifo_read_pos = (fifo_read_pos + taken) % AUDIO_FIFO_SAMPLES;
    fifo_count -= taken;
    *overrun_generation = fifo_overrun_events;

    xSemaphoreGive(fifo_mutex);
    return taken;
}

/**
 * Check whether a recognized label is a valid fan-control command.
 *
 * Background and unknown labels are intentionally excluded.
 */
static bool is_actionable_command(const char *command) {
    return strcmp(command, "turn_on") == 0 || 
           strcmp(command, "turn_off") == 0 || 
           strcmp(command, "low") == 0 || 
           strcmp(command, "medium") == 0 || 
           strcmp(command, "high") == 0 || 
           strcmp(command, "max") == 0;
}

/**
 * Initialize the fan-control application and run continuous microphone-to-KWS
 * processing. Accepted voice commands are forwarded to the PI controller.
 */
void app_main(void) {
    ESP_LOGI(TAG, "===== CONTINUOUS MIC -> FIFO -> KWS -> PI =====");

    esp_err_t ret = ESP_OK;
    bool capture_started = false;
#if KWS_MOTOR_ENABLED
    bool control_started = false;
#endif

    if (KWS_ACTIVE_SAMPLES == 0 || KWS_SLIDE_SAMPLES == 0 || KWS_ACTIVE_SAMPLES > KWS_WINDOW_SAMPLES || AUDIO_FIFO_SAMPLES < MIC_READ_SAMPLES) {
        ESP_LOGE(TAG, "Invalid KWS / FIFO configuration");
        return;
    }

#if KWS_MOTOR_ENABLED
    ret = control_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Control initialization failed: %s", esp_err_to_name(ret));
        goto cleanup;
    }
    control_started = true;
#endif

    ret = preprocess_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Preprocessing initialization failed: %s", esp_err_to_name(ret));
        goto cleanup;
    }

    ret = command_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "KWS initialization failed: %s", esp_err_to_name(ret));
        goto cleanup;
    }

    // Keep the proven 32000-sample preprocessing input unchanged.
    // Allocate the larger PCM block FIRST to reduce heap fragmentation.
    ESP_LOGI(TAG, "Heap before allocation: free=%u, largest=%u", (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT), (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

    pcm_window = heap_caps_malloc((size_t)KWS_WINDOW_SAMPLES * sizeof(int16_t), MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "PCM: %u bytes, allocated=%d, largest=%u", (unsigned)((size_t)KWS_WINDOW_SAMPLES * sizeof(int16_t)), pcm_window != NULL, (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    if (pcm_window == NULL) {
        ESP_LOGE(TAG, "Cannot allocate the 2-second PCM window");
        goto cleanup;
    }

    audio_fifo = heap_caps_malloc((size_t)AUDIO_FIFO_SAMPLES * sizeof(int16_t), MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "FIFO: %u bytes, allocated=%d, largest=%u", (unsigned)((size_t)AUDIO_FIFO_SAMPLES * sizeof(int16_t)), audio_fifo != NULL, (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));
    if (audio_fifo == NULL) {
        ESP_LOGE(TAG, "Cannot allocate the 1-second audio FIFO");
        goto cleanup;
    }

    fifo_mutex = xSemaphoreCreateMutex();
    capture_done = xSemaphoreCreateBinary();
    if (fifo_mutex == NULL || capture_done == NULL) {
        ESP_LOGE(TAG, "Cannot allocate audio synchronization objects");
        goto cleanup;
    }

#if KWS_OLED_ENABLED
    display_init();
    display_clear();
    vTaskDelay(pdMS_TO_TICKS(3000));
    oled_initialized = true;
    oled_draw_status("WAIT", "MIC START", -1.0f, true, false);
    ESP_LOGI(TAG, "OLED initialized");
#endif

    ret = audio_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Microphone initialization failed: %s", esp_err_to_name(ret));
        goto cleanup;
    }

    // Reduce preprocessing log spam without changing feature extraction.
    esp_log_level_set("PREPROCESS", ESP_LOG_WARN);

    kws_task_handle = xTaskGetCurrentTaskHandle();
    capture_error = ESP_OK;
    capture_enabled = true;

    BaseType_t created = xTaskCreate(audio_capture_task, "audio_capture", AUDIO_TASK_STACK_BYTES, NULL, AUDIO_TASK_PRIORITY, NULL);
    if (created != pdPASS) {
        capture_enabled = false;
        ESP_LOGE(TAG, "Cannot create microphone capture task");
        goto cleanup;
    }
    capture_started = true;

    ESP_LOGI(TAG, "Continuous microphone capture started");
#if KWS_OLED_ENABLED
    oled_draw_status("LISTENING", "MIC READY", -1.0f, true, false);
#endif
    ESP_LOGI(TAG, "KWS input=%u samples; feature coverage=%u (%.3f s)", (unsigned)KWS_WINDOW_SAMPLES, (unsigned)KWS_ACTIVE_SAMPLES, (double)KWS_ACTIVE_SAMPLES / KWS_SAMPLE_RATE);
    ESP_LOGI(TAG, "FIFO=%u samples; slide=%u samples; motor_enabled=%d", (unsigned)AUDIO_FIFO_SAMPLES, (unsigned)KWS_SLIDE_SAMPLES, KWS_MOTOR_ENABLED);

    // app_main owns pcm_window. The audio task never modifies it.
    size_t pcm_filled = 0;
    bool has_full_window = false;
    uint32_t last_overrun_generation = 0;
    int last_accepted_class = -1;
    TickType_t last_accepted_tick = 0;

    while (true) {
        // Allow IDLE0 to run between KWS iterations and feed the task WDT.
        // This also covers paths that return to the loop via 'continue'.
        vTaskDelay(pdMS_TO_TICKS(10));

        if (capture_error != ESP_OK) {
            ESP_LOGE(TAG, "Capture stopped unexpectedly");
#if KWS_OLED_ENABLED
            oled_draw_status("ERROR", "MIC FAILED", -1.0f, true, false);
#endif
            break;
        }

        uint32_t overrun_generation = 0;
        uint32_t dropped_samples = 0;
        const size_t available = fifo_available(&overrun_generation, &dropped_samples);

        // Never combine PCM segments separated by FIFO data loss.
        if (overrun_generation != last_overrun_generation) {
            ESP_LOGW(TAG, "Audio FIFO overrun: events=%u, dropped=%u samples; resync", (unsigned)overrun_generation, (unsigned)dropped_samples);
            last_overrun_generation = overrun_generation;
            pcm_filled = 0;
            has_full_window = false;
#if KWS_OLED_ENABLED
            oled_draw_status("WAIT", "FIFO RESYNC", -1.0f, true, false);
#endif
        }

        if (!has_full_window) {
            // Assemble the first complete 2-second window incrementally.
            if (available == 0) {
                ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1000));
                continue;
            }

            const size_t room = KWS_WINDOW_SAMPLES - pcm_filled;
            const size_t wanted = (available < room) ? available : room;
            uint32_t pop_generation = 0;
            const size_t taken = fifo_pop(pcm_window + pcm_filled, wanted, &pop_generation);

            if (pop_generation != last_overrun_generation) {
                last_overrun_generation = pop_generation;
                pcm_filled = 0;
                has_full_window = false;
                continue;
            }

            pcm_filled += taken;
            if (pcm_filled < KWS_WINDOW_SAMPLES) {
                continue;
            }

            has_full_window = true;
            ESP_LOGI(TAG, "First full audio window ready");
#if KWS_OLED_ENABLED
            oled_draw_status("PROCESSING", "FIRST WINDOW", -1.0f, true, false);
#endif
        } 
        else {
            // Allow at least 250 ms of new audio before the next inference.
            if (available < KWS_SLIDE_SAMPLES) {
                ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(1000));
                continue;
            }

            // Consume all currently queued samples to keep inference current.
            // If inference takes >250 ms, older intermediate windows are skipped.
            const size_t advance = (available < KWS_WINDOW_SAMPLES) ? available : KWS_WINDOW_SAMPLES;
            memmove(pcm_window, pcm_window + advance, (KWS_WINDOW_SAMPLES - advance) * sizeof(int16_t));

            uint32_t pop_generation = 0;
            const size_t taken = fifo_pop(pcm_window + (KWS_WINDOW_SAMPLES - advance), advance, &pop_generation);
            if (pop_generation != last_overrun_generation || taken != advance) {
                last_overrun_generation = pop_generation;
                pcm_filled = 0;
                has_full_window = false;
                continue;
            }
        }
        // Measure the complete command processing response time.
        int64_t response_start_us = esp_timer_get_time();
        // preprocess.c and command.cpp receive the same input format as before.
        ret = preprocess_extract_pcm16((const uint8_t *)pcm_window, KWS_WINDOW_SAMPLES, features, sizeof(features));
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Feature extraction failed: %s", esp_err_to_name(ret));
            break;
        }

        int predicted_class = -1;
        float confidence = 0.0f;
        uint32_t inference_time_us = 0;
        ret = command_predict(features, sizeof(features), &predicted_class, &confidence, &inference_time_us);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "KWS inference failed: %s", esp_err_to_name(ret));
            break;
        }

        if (predicted_class < 0 || predicted_class >= KWS_NUM_CLASSES) {
            ESP_LOGW(TAG, "Invalid predicted class: %d", predicted_class);
            continue;
        }

        const char *command = KWS_LABELS[predicted_class];
        ESP_LOGI(TAG, "KWS: %s | Confidence: %.4f | Inference: %.2f ms", command, (double)confidence, (double)inference_time_us / 1000.0);
        if (!is_actionable_command(command)) {
#if KWS_OLED_ENABLED
            oled_show_listening(command, confidence);
#endif
            continue;
        }
        if (confidence < KWS_CONFIDENCE_MIN ) {
#if KWS_OLED_ENABLED
            oled_show_rejected(command, confidence);
#endif
            continue;
        }

        TickType_t now = xTaskGetTickCount();
        if (predicted_class == last_accepted_class && (TickType_t)(now - last_accepted_tick) < pdMS_TO_TICKS(KWS_COMMAND_COOLDOWN_MS)) {
#if KWS_OLED_ENABLED
            oled_show_listening(command, confidence);
#endif
            continue;
        }

#if KWS_MOTOR_ENABLED
        ret = control_set_command(command);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to execute command: %s", command);
            continue;
        }
        ESP_LOGI(TAG, "Command executed: %s", command);
#else

        ESP_LOGI(TAG, "TEST ONLY: recognized command = %s", command);
#endif

int64_t response_end_us = esp_timer_get_time();
float response_time_ms = (float)(response_end_us - response_start_us) / 1000.0f;
ESP_LOGI(TAG, "Response time: %.2f ms", response_time_ms);

#if KWS_OLED_ENABLED
    // Save the accepted command before refreshing the OLED.
    snprintf(oled_last_command, sizeof(oled_last_command), "%s", command);
    oled_draw_status("ACCEPTED", command, confidence, true, true);
#endif
        last_accepted_class = predicted_class;
        last_accepted_tick = now;
    }

cleanup:
#if KWS_OLED_ENABLED
    if (oled_initialized) {
        if (ret != ESP_OK || capture_error != ESP_OK) {
            oled_draw_status("ERROR", "APP STOP", -1.0f, true, false);
        } 
        else {
            oled_draw_status("WAIT", "APP STOP", -1.0f, true, false);
        }
    }
#endif

#if KWS_MOTOR_ENABLED
    if (control_started) {
        // This is a software stop, not an emergency power disconnect.
        control_set_command("turn_off");
    }
#endif

    if (capture_started) {
        capture_enabled = false;
        // audio_read has a finite timeout; wait for the producer to exit.
        xSemaphoreTake(capture_done, portMAX_DELAY);
    }

    if (capture_done != NULL) {
        vSemaphoreDelete(capture_done);
        capture_done = NULL;
    }
    if (fifo_mutex != NULL) {
        vSemaphoreDelete(fifo_mutex);
        fifo_mutex = NULL;
    }
    if (audio_fifo != NULL) {
        heap_caps_free(audio_fifo);
        audio_fifo = NULL;
    }
    if (pcm_window != NULL) {
        heap_caps_free(pcm_window);
        pcm_window = NULL;
    }

    ESP_LOGE(TAG, "Continuous KWS application stopped");
}
