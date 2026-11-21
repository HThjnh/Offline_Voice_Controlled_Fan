#ifndef AUDIO_H
#define AUDIO_H

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

/**
 * @brief Initialize the I2S microphone interface.
 */
esp_err_t audio_init(void);

/**
 * @brief Read audio samples from the microphone.
 *
 * @param buffer Destination buffer for audio samples.
 * @param buffer_size Buffer size in bytes.
 * @param bytes_read Number of bytes actually read.
 */
esp_err_t audio_read(int32_t *buffer, size_t buffer_size, size_t *bytes_read);

#endif 