#ifndef COMMAND_H
#define COMMAND_H

#include "esp_err.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the TensorFlow Lite Micro keyword spotting component.
 *
 * Loads the embedded model, registers its operators, allocates the
 * tensor arena, and validates the input/output tensor configuration.
 * Repeated calls after successful initialization return ESP_OK.
 *
 * @return ESP_OK on success, or an ESP-IDF error code on failure.
 */
esp_err_t command_init(void);

/**
 * @brief Predict a voice command from quantized Log-Mel features.
 *
 * The input must contain KWS_N_FRAMES * KWS_N_MEL signed INT8 values
 * in the feature order expected by the trained model.
 * Call command_init() before invoking this function.
 *
 * @param features Pointer to the quantized Log-Mel feature buffer.
 * @param feature_count Number of INT8 feature values in the buffer.
 * @param predicted_class Output pointer for the predicted class index.
 * @param confidence Output pointer for the calculated confidence score.
 * @return ESP_OK on success, or an ESP-IDF error code on failure.
 */
esp_err_t command_predict(const int8_t *features, size_t feature_count, int *predicted_class, float *confidence);

#ifdef __cplusplus
}
#endif



#endif