#include "encoder.h"
#include "driver/pulse_cnt.h"
#include "esp_log.h"
#include "esp_timer.h"

#define ENCODER_GPIO_A  32
#define ENCODER_GPIO_B  33
#define PCNT_HIGH_LIMIT 32767
#define PCNT_LOW_LIMIT -32768

static const char *TAG = "ENCODER";

static pcnt_unit_handle_t encoder_unit = NULL;

/**
 * @brief Initialize PCNT for bidirectional encoder counting.
 *
 * Counts rising edges of encoder channel A.
 * Channel B determines whether the counter
 * increments or decrements.
 *
 * Uses x1 quadrature decoding.
 */
esp_err_t encoder_init(void) {
    if(encoder_unit != NULL) {
        return ESP_OK;
    }
    // Configure the pulse counter limits
    pcnt_unit_config_t unit_config = {
        .high_limit = PCNT_HIGH_LIMIT,
        .low_limit  = PCNT_LOW_LIMIT,
        .flags = {
            .accum_count = 1
        }
    };
    pcnt_unit_handle_t unit = NULL;

    esp_err_t ret = pcnt_new_unit(&unit_config, &unit);
    if(ret != ESP_OK) {
        return ret;
    }

    // Use encoder A for pulse counting and B for direction
    pcnt_chan_config_t channel_config = {
        .edge_gpio_num  = ENCODER_GPIO_A,
        .level_gpio_num = ENCODER_GPIO_B
    };
    pcnt_channel_handle_t channel = NULL;
    ret = pcnt_new_channel(unit, &channel_config, &channel);
    if(ret != ESP_OK) {
        pcnt_del_unit(unit);
        return ret;
    }

    // Count only rising edges of encoder A
    ret = pcnt_channel_set_edge_action(channel, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_HOLD);
    if(ret != ESP_OK) {
        pcnt_del_channel(channel);
        pcnt_del_unit(unit);
        return ret;
    }

    // Reverse the counting direction when encoder B is LOW
    ret = pcnt_channel_set_level_action(channel, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE);
    if (ret != ESP_OK) {
        pcnt_del_channel(channel);
        pcnt_del_unit(unit);
        return ret;
    }

    // Filter short glitches on the encoder input
    pcnt_glitch_filter_config_t filter_config = {
        .max_glitch_ns = 1000
    };
    ret = pcnt_unit_set_glitch_filter(unit, &filter_config);
    if(ret != ESP_OK) {
        pcnt_del_channel(channel);
        pcnt_del_unit(unit);
        return ret;
    }
    // Accumulate positive and negative counter overflows
    ret = pcnt_unit_add_watch_point(unit, PCNT_HIGH_LIMIT);
    if (ret != ESP_OK) {
        pcnt_del_channel(channel);
        pcnt_del_unit(unit);
        return ret;
    }

    ret = pcnt_unit_add_watch_point(unit, PCNT_LOW_LIMIT);
    if (ret != ESP_OK) {
        pcnt_unit_remove_watch_point(unit, PCNT_HIGH_LIMIT);
        pcnt_del_channel(channel);
        pcnt_del_unit(unit);
        return ret;
    }

    // Enable the PCNT unit
    ret = pcnt_unit_enable(unit);
    if(ret != ESP_OK) {
        pcnt_unit_remove_watch_point(unit, PCNT_HIGH_LIMIT);
        pcnt_unit_remove_watch_point(unit, PCNT_LOW_LIMIT);
        pcnt_del_channel(channel);
        pcnt_del_unit(unit);
        return ret;
    }

    // Clear the initial counter value
    ret = pcnt_unit_clear_count(unit);
    if(ret == ESP_OK) {
        ret = pcnt_unit_start(unit);
    }
    if(ret != ESP_OK) {
        // Disable PCNT before releasing resources
        pcnt_unit_disable(unit);
        // Remove configured watch points
        pcnt_unit_remove_watch_point(unit, PCNT_HIGH_LIMIT);
        pcnt_unit_remove_watch_point(unit, PCNT_LOW_LIMIT);
        // Release PCNT resources
        pcnt_del_channel(channel);
        pcnt_del_unit(unit);
        return ret;
    }

    encoder_unit = unit;
    ESP_LOGI(TAG, "Encoder initialized (PCNT x1, bidirectional)");
    return ESP_OK;
}

/**
 * @brief Read the current encoder count.
 */
esp_err_t encoder_get_count(int *count) {
    if(encoder_unit == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    if(count == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    return pcnt_unit_get_count(encoder_unit, count);
}

/**
 * @brief Reset the PCNT count to zero.
 */
esp_err_t encoder_reset(void) {
    if(encoder_unit == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    return pcnt_unit_clear_count(encoder_unit);
}


/**
 * @brief Calculate signed RPM from encoder count differences.
 *
 * Uses actual elapsed time in microseconds rather than
 * assuming that the task executes at an exact interval.
 */
esp_err_t encoder_get_rpm(encoder_speed_t *speed, float counts_per_rev, float *rpm) {
    if (speed == NULL || rpm == NULL || counts_per_rev <= 0.0f) {
        return ESP_ERR_INVALID_ARG;
    }

    int current_count = 0;
    esp_err_t ret = encoder_get_count(&current_count);
    if (ret != ESP_OK) {
        return ret;
    }

    int64_t current_time_us = esp_timer_get_time();
    // Initialize the first measurement
    if (!speed->initialized) {
        speed->previous_count = current_count;
        speed->previous_time_us = current_time_us;
        speed->initialized = true;

        *rpm = 0.0f;
        return ESP_OK;
    }

    // Calculate elapsed time
    int64_t delta_time_us = current_time_us - speed->previous_time_us;

    if (delta_time_us <= 0) {
        return ESP_ERR_INVALID_STATE;
    }

    // Calculate signed encoder count difference
    int64_t delta_count = (int64_t)current_count - speed->previous_count;

    // Convert counts per interval into RPM
    *rpm = (float)( ((double)delta_count * 60.0 * 1000000.0) / ((double)counts_per_rev * (double)delta_time_us));

    // Update measurement history
    speed->previous_count = current_count;
    speed->previous_time_us = current_time_us;

    return ESP_OK;
}



