#ifndef FAN_H
#define FAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    SPEED_OFF,
    SPEED_LOW,
    SPEED_MEDIUM,
    SPEED_HIGH,
    SPEED_MAX
} fan_level_t;

void fan_init();                           //LEDC Timer, LEDC Channel, GPIO PWM configuration
void fan_set_pwm(uint16_t duty);

#ifdef __cplusplus
}
#endif

#endif