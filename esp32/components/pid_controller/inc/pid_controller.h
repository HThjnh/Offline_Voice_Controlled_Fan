#ifndef PID_CONTROLLER_H
#define PID_CONTROLLER_H

#include <stdbool.h>
#include <stdint.h>

/**
 * @brief PID controller state and configuration.
 *
 * Gains are expressed using RPM as the input
 * and PWM duty counts as the output.
 */
typedef struct {
    float kp;
    float ki;
    float kd;

    float dt;
    float target_rpm;

    float integral;
    float prev_rpm;
    bool has_prev_rpm;

    float output_min;
    float output_max;
} pid_controller_t;

/**
 * @brief Initialize a PID controller.
 *
 * @param pid Pointer to the controller.
 * @param kp Proportional gain.
 * @param ki Integral gain.
 * @param kd Derivative gain.
 * @param dt Control period in seconds.
 */
void pid_controller_init(pid_controller_t *pid, float kp, float ki, float kd, float dt);

/**
 * @brief Set the desired motor speed in RPM.
 */
void pid_controller_set_target(pid_controller_t *pid, float target_rpm);

/**
 * @brief Compute the PWM duty from measured RPM.
 *
 * Call this function at a fixed sampling period.
 *
 * @param pid Pointer to the PID controller.
 * @param actual_rpm Measured motor speed in RPM.
 * @return PWM duty in the range 0 to 1000.
 */
uint16_t pid_controller_compute( pid_controller_t *pid, float actual_rpm);

/**
 * @brief Clear the integral and derivative history.
 *
 * @param pid Pointer to the PID controller.
 */
void pid_controller_reset(pid_controller_t *pid);

#endif