#include "control.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "fan.h"
#include "encoder.h"
#include "pid_controller.h"

#include "esp_log.h"
#include "esp_err.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

// Previously tested control parameters
#define CONTROL_CPR                     228.4f
#define CONTROL_DT                      0.1f
#define CONTROL_PERIOD_MS               100
#define CONTROL_LOG_INTERVAL_MS         1000
#define CONTROL_COMMAND_TIMEOUT_MS      0

// Allow the motor to accelerate before stall detection.
#define CONTROL_STARTUP_GRACE_MS        2000

// Stall detection parameters.
#define CONTROL_STALL_RPM               5.0f
#define CONTROL_STALL_SAMPLES           5

#define CONTROL_KP                      0.5f
#define CONTROL_KI                      3.0f
#define CONTROL_KD                      0.0f

#define CONTROL_PWM_MAX                 1000.0f
#define CONTROL_START_I                 600.0f

// Target RPM values for each voice command
// These values must be validated experimentally
#define RPM_TURN_ON                     80.0f
#define RPM_TURN_OFF                    0.0f
#define RPM_LOW                         110.0f
#define RPM_MEDIUM                      140.0f
#define RPM_HIGH                        170.0f
#define RPM_MAX                         200.0f

static const char *TAG = "CONTROL";

// Target speed updated by the KWS system
static float target_rpm = 0.0f;
// Last valid KWS command timestamp; protected by target_mutex.
static TickType_t last_command_tick = 0;

// Synchronization between the KWS and control tasks
static SemaphoreHandle_t target_mutex = NULL;
static TaskHandle_t control_task_handle = NULL;
// Forward declaration of the PI control task
static void control_task(void *arg);

// --------------------------------------------------
// Initialize the motor control system
// --------------------------------------------------
esp_err_t control_init(void)
{
    // Prevent multiple task instances
    if (control_task_handle != NULL) {
        return ESP_OK;
    }
    // Initialize the encoder
    esp_err_t ret = encoder_init();
    if (ret != ESP_OK) {
        return ret;
    }
    // Reset the encoder pulse counter
    ret = encoder_reset();
    if (ret != ESP_OK) {
        return ret;
    }
    // Keep routine FAN logs from flooding the serial monitor.
    // Warnings and errors are still shown.
    esp_log_level_set("FAN", ESP_LOG_WARN);
    // Initialize the motor driver
    fan_init();
    fan_set_pwm(0);

    // Create a mutex for target RPM access
    if (target_mutex == NULL) {
        target_mutex = xSemaphoreCreateMutex();
    }
    if (target_mutex == NULL) {
        return ESP_ERR_NO_MEM;
    }

    target_rpm = 0.0f;
    last_command_tick = xTaskGetTickCount();
    // Create the dedicated PI control task
    BaseType_t result = xTaskCreate( control_task, "control_task", 4096, NULL, 5, &control_task_handle);

    if (result != pdPASS) {
        control_task_handle = NULL;
        fan_set_pwm(0);
        ESP_LOGE(TAG, "Cannot create control task");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Control initialized");
    ESP_LOGI(TAG, "Motor OFF - waiting for KWS");

    return ESP_OK;
}

// --------------------------------------------------
// Read the target RPM in a thread-safe manner
// --------------------------------------------------
float control_get_target_rpm(void) {
    float value = 0.0f;
    if (target_mutex == NULL) {
        return 0.0f;
    }
    if (xSemaphoreTake(target_mutex, portMAX_DELAY) == pdTRUE) {
        value = target_rpm;
        xSemaphoreGive(target_mutex);
    }
    return value;
}

// --------------------------------------------------
// Process a voice command from the KWS system
// --------------------------------------------------
esp_err_t control_set_command(const char *command) {
    if (command == NULL) {
        return ESP_ERR_INVALID_ARG;
    }
    if (target_mutex == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    float new_target = 0.0f;
    if (strcmp(command, "turn_on") == 0) {
        new_target = RPM_TURN_ON;
    }
    else if (strcmp(command, "low") == 0) {
        new_target = RPM_LOW;
    }
    else if (strcmp(command, "medium") == 0) {
        new_target = RPM_MEDIUM;
    }
    else if (strcmp(command, "high") == 0) {
        new_target = RPM_HIGH;
    }
    else if (strcmp(command, "max") == 0) {
        new_target = RPM_MAX;
    }
    else if (strcmp(command, "turn_off") == 0) {
        new_target = RPM_TURN_OFF;
    }
    else {
        // Ignore unknown or background commands
        // Keep the current target speed unchanged
        return ESP_ERR_NOT_FOUND;
    }

    if (xSemaphoreTake(target_mutex, portMAX_DELAY) != pdTRUE) {
        return ESP_FAIL;
    }

    target_rpm = new_target;
    // A new valid command renews the five-second test timeout.
    last_command_tick = xTaskGetTickCount();
    xSemaphoreGive(target_mutex);

    ESP_LOGI(TAG, "Command: %s | Target: %.1f RPM", command, new_target);
    return ESP_OK;
}

// --------------------------------------------------
// PI control task executed every 100 milliseconds
// --------------------------------------------------
static void control_task(void *arg) {
    pid_controller_t pid;

    // Initialize the PI controller
    pid_controller_init(
        &pid,
        CONTROL_KP,
        CONTROL_KI,
        CONTROL_KD,
        CONTROL_DT
    );

    // Limit the maximum PWM output
    pid.output_max = CONTROL_PWM_MAX;

    encoder_speed_t speed = {0};

    bool motor_active = false;
    float previous_target = 0.0f;

    int stalled_samples = 0;

    TickType_t last_wake = xTaskGetTickCount();
    TickType_t last_log_tick = last_wake;
    TickType_t motor_start_tick = last_wake;

    while (1) {
        // Maintain a fixed control period
        vTaskDelayUntil( &last_wake, pdMS_TO_TICKS(CONTROL_PERIOD_MS));
        float target = 0.0f;
        bool command_timed_out = false;

        // Read the command and enforce the timeout under the same mutex.
        if (xSemaphoreTake(target_mutex, portMAX_DELAY) == pdTRUE) {
            target = target_rpm;

            if (CONTROL_COMMAND_TIMEOUT_MS > 0 && target > 0.0f && (TickType_t)(xTaskGetTickCount() - last_command_tick) >= pdMS_TO_TICKS(CONTROL_COMMAND_TIMEOUT_MS)) {
                target_rpm = 0.0f;
                target = 0.0f;
                command_timed_out = true;
            }
            xSemaphoreGive(target_mutex);
        }

        if (command_timed_out) {
            ESP_LOGW(TAG, "Command timeout (5 s) - stopping motor");
        }
        // ------------------------------------------
        // Stop the motor when target RPM is zero
        // ------------------------------------------
        if (target <= 0.0f) {
            if (motor_active) {
                fan_set_pwm(0);
                pid_controller_set_target(&pid, 0.0f);
                motor_active = false;
                previous_target = 0.0f;
                stalled_samples = 0;
                ESP_LOGI(TAG, "Motor STOP");
            }
            continue;
        }
        // ------------------------------------------
        // Initialize the motor startup sequence
        // ------------------------------------------
        if (!motor_active) {
            // Reset the PI controller state
            pid_controller_reset(&pid);

            // Set the requested speed
            pid_controller_set_target(&pid, target);

            // Apply a temporary startup integral value
            // to provide sufficient initial PWM
            pid.integral = CONTROL_START_I;

            // Reset the RPM measurement state
            speed = (encoder_speed_t){0};

            float initial_rpm = 0.0f;

            esp_err_t ret = encoder_get_rpm( &speed, CONTROL_CPR, &initial_rpm);
            if (ret != ESP_OK) {
                fan_set_pwm(0);
                ESP_LOGE(TAG, "Encoder init RPM error");
                continue;
            }

            // Calculate the initial PWM command
            uint16_t pwm = pid_controller_compute( &pid, 0.0f);

            // Start the motor
            fan_set_pwm(pwm);

            motor_active = true;
            previous_target = target;
            stalled_samples = 0;
            last_log_tick = xTaskGetTickCount();
            motor_start_tick = xTaskGetTickCount();

            ESP_LOGI(TAG, "Motor START | Target=%.1f | PWM=%u", target, (unsigned)pwm);
            continue;
        }

        // ------------------------------------------
        // Update the target speed when it changes
        // ------------------------------------------
        if (target != previous_target) {
            pid_controller_set_target(&pid, target);
            previous_target = target;
            ESP_LOGI(TAG, "New target: %.1f RPM", target);
        }

        // ------------------------------------------
        // Read the actual RPM from the encoder
        // ------------------------------------------
        float signed_rpm = 0.0f;

        esp_err_t ret = encoder_get_rpm( &speed, CONTROL_CPR, &signed_rpm);
        if (ret != ESP_OK) {
            // Stop the motor if encoder reading fails
            fan_set_pwm(0);

            if (xSemaphoreTake(target_mutex, portMAX_DELAY) == pdTRUE) {
                target_rpm = 0.0f;
                xSemaphoreGive(target_mutex);
            }

            motor_active = false;

            ESP_LOGE(TAG, "Encoder RPM error");
            continue;
        }

        // Use the absolute RPM for one-direction control
        float actual_rpm = signed_rpm;

        if (actual_rpm < 0.0f) {
            actual_rpm = -actual_rpm;
        }

        // ------------------------------------------
        // Detect a stalled or stopped motor
        // ------------------------------------------
        // Wait until the motor startup grace period has elapsed.
        const TickType_t check_tick = xTaskGetTickCount();

        const bool startup_complete =
            (TickType_t)(check_tick - motor_start_tick) >=
            pdMS_TO_TICKS(CONTROL_STARTUP_GRACE_MS);

        if (!startup_complete) {
            stalled_samples = 0;
        }
        else if (actual_rpm < CONTROL_STALL_RPM) {
            stalled_samples++;
        }
        else {
            stalled_samples = 0;
        }

        // Stop the motor after five consecutive
        // low-speed measurements
        if (stalled_samples >= CONTROL_STALL_SAMPLES) {
            fan_set_pwm(0);

            if (xSemaphoreTake(target_mutex, portMAX_DELAY) == pdTRUE) {
                target_rpm = 0.0f;
                xSemaphoreGive(target_mutex);
            }

            pid_controller_set_target(&pid, 0.0f);

            motor_active = false;
            previous_target = 0.0f;
            stalled_samples = 0;

            ESP_LOGW(TAG, "Motor stalled - stopped");
            continue;
        }

        // ------------------------------------------
        // Calculate the PWM using the PI controller
        // ------------------------------------------
        uint16_t pwm = pid_controller_compute( &pid, actual_rpm);

        // Apply the new PWM duty cycle
        fan_set_pwm(pwm);

        // Print the control status every second, not every 100 ms.
        TickType_t now = xTaskGetTickCount();
        if ((TickType_t)(now - last_log_tick) >= pdMS_TO_TICKS(CONTROL_LOG_INTERVAL_MS)) {
            ESP_LOGI(TAG, "Target=%.1f | RPM=%.2f | PWM=%u", target, actual_rpm, (unsigned)pwm);
            last_log_tick = now;
        }
    }
}


