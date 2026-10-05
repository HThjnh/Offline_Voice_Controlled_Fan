#ifndef FAN_H
#define FAN_H

typedef enum {
    SPEED_OFF,
    SPEED_LOW,
    SPEED_MEDIUM,
    SPEED_HIGH,
    SPEED_MAX
} fan_level_t;
void fan_init();                           //LEDC Timer, LEDC Channel, GPIO PWM configuration
void fan_set_level(fan_level_t level);     //Level of fan
void fan_stop(void);                           //Turn off fan

#endif