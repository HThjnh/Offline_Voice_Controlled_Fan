
#include "pid_controller.h"
#include <stddef.h>

#define PID_PWM_MIN 0.0f
#define PID_PWM_MAX 1023.0f

static float clamp_float(float value, float min, float max) {
    if (value < min) {
        return min;
    }

    if (value > max) {
        return max;
    }

    return value;
}

/**
 * @brief Initialize PID gains and internal state.
 */
void pid_controller_init(pid_controller_t *pid, float kp, float ki, float kd, float dt) {
    if (pid == NULL) {
        return;
    }

    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;

    pid->dt = (dt > 0.0f) ? dt : 0.1f;
    pid->target_rpm = 0.0f;

    pid->output_min = PID_PWM_MIN;
    pid->output_max = PID_PWM_MAX;

    pid_controller_reset(pid);
}

/**
 * @brief Clear accumulated PID state.
 */
void pid_controller_reset(pid_controller_t *pid) {
    if (pid == NULL) {
        return;
    }

    pid->integral = 0.0f;
    pid->prev_rpm = 0.0f;
    pid->has_prev_rpm = false;
}

/**
 * @brief Update the target motor speed.
 */
void pid_controller_set_target(pid_controller_t *pid, float target_rpm) {
    if (pid == NULL) {
        return;
    }

    if (target_rpm <= 0.0f) {
        pid->target_rpm = 0.0f;
        pid_controller_reset(pid);
        return;
    }

    pid->target_rpm = target_rpm;
}

/**
 * @brief Calculate PWM using discrete PID control.
 *
 * Includes conditional integration to prevent
 * integral windup when PWM reaches its limits.
 *
 * Uses derivative on measurement to avoid
 * derivative kick when the setpoint changes.
 */
uint16_t pid_controller_compute(pid_controller_t *pid, float actual_rpm) {
    if (pid == NULL || pid->dt <= 0.0f) {
        return 0;
    }

    // Stop the motor and clear PID state
    if (pid->target_rpm <= 0.0f) {
        pid_controller_reset(pid);
        return 0;
    }

    // Calculate speed error
    float error = pid->target_rpm - actual_rpm;

    // Proportional term
    float p_term = pid->kp * error;

    // Derivative on measurement
    float d_term = 0.0f;

    if (pid->has_prev_rpm) {
        float rpm_rate =
            (actual_rpm - pid->prev_rpm) / pid->dt;

        d_term = -pid->kd * rpm_rate;
    }

    // Calculate candidate integral value
    float candidate_integral = pid->integral + pid->ki * error * pid->dt;

    // Calculate candidate PWM
    float candidate_output = p_term + candidate_integral + d_term;

    // Anti-windup: prevent further saturation
    bool saturating_high = candidate_output > pid->output_max && error > 0.0f;

    bool saturating_low = candidate_output < pid->output_min && error < 0.0f;

    if (!saturating_high && !saturating_low) {
        pid->integral = candidate_integral;
    }

    // Calculate and limit the final PWM output
    float output = p_term + pid->integral + d_term;

    output = clamp_float(output, pid->output_min, pid->output_max);

    // Update derivative history
    pid->prev_rpm = actual_rpm;
    pid->has_prev_rpm = true;

    return (uint16_t)(output + 0.5f);
}
