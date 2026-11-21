#ifndef ENCODER_H
#define ENCODER_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @brief Initialize the encoder pulse counter.
 *
 * Configures ESP32 PCNT to count rising edges
 * from encoder channel A (x1 decoding).
 *
 * @return ESP_OK on success, otherwise an error code.
 */
esp_err_t encoder_init(void);

/**
 * @brief Read the current encoder pulse count.
 *
 * @param count Pointer to store the current count.
 * @return ESP_OK on success, otherwise an error code.
 */
esp_err_t encoder_get_count(int *count);

/**
 * @brief Reset the encoder pulse counter to zero.
 *
 * @return ESP_OK on success, otherwise an error code.
 */
esp_err_t encoder_reset(void);


/**
 * @brief State for encoder speed measurement.
 */
typedef struct {
    int previous_count;
    int64_t previous_time_us;
    bool initialized;
} encoder_speed_t;

/**
 * @brief Calculate motor output shaft RPM.
 *
 * Uses signed encoder count differences and actual
 * elapsed time between consecutive measurements.
 *
 * @param speed Speed measurement state.
 * @param counts_per_rev Calibrated x1 counts per output revolution.
 * @param rpm Pointer to the calculated signed RPM.
 *
 * @return ESP_OK on success, otherwise an error code.
 */
esp_err_t encoder_get_rpm(
    encoder_speed_t *speed,
    float counts_per_rev,
    float *rpm
);


#ifdef __cplusplus
}
#endif

#endif