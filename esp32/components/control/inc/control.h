
#ifndef CONTROL_H
#define CONTROL_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the motor speed control system.
 *
 * Initializes the encoder, motor driver, PI controller,
 * and creates a dedicated FreeRTOS control task.
 *
 * The control task runs independently of the KWS system
 * and continuously regulates the motor speed.
 *
 * @return ESP_OK on success, otherwise an error code.
 */
esp_err_t control_init(void);

/**
 * @brief Process a voice command from the KWS system.
 *
 * Converts a recognized voice command into a target RPM.
 *
 * Supported commands:
 * - "turn_on"
 * - "turn_off"
 * - "low"
 * - "medium"
 * - "high"
 * - "max"
 *
 * Unknown or background commands are ignored.
 *
 * @param command Null-terminated command string.
 * @return ESP_OK if the command is valid,
 *         ESP_ERR_INVALID_ARG if command is NULL,
 *         or an error code if processing fails.
 */
esp_err_t control_set_command(const char *command);

/**
 * @brief Get the current target motor speed.
 *
 * Reads the target RPM using a mutex to ensure
 * safe access between the KWS and control tasks.
 *
 * @return Current target speed in RPM.
 *         Returns 0.0f if the control system
 *         has not been initialized.
 */
float control_get_target_rpm(void);

#ifdef __cplusplus
}
#endif

#endif // CONTROL_H
