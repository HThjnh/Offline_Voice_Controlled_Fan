#ifndef FAN_H
#define FAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif 

/**
 * @brief Initialize the TB6612FNG motor driver and PWM output.
 */
void fan_init(void);                           //LEDC Timer, LEDC Channel, GPIO PWM configuration

/**
 * @brief Set the motor PWM duty cycle.
 *
 * @param duty PWM duty cycle (0 to 1023).
 */
void fan_set_pwm(uint16_t duty);

#ifdef __cplusplus
}
#endif

#endif