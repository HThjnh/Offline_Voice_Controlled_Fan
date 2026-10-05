#include "fan.h"
#include "driver/ledc.h"

#define FAN_PWM_GPIO        18
#define FAN_PWM_FREQ        20000
#define FAN_PWM_RESOLUTION  LEDC_TIMER_10_BIT
#define FAN_PWM_TIMER       LEDC_TIMER_0
#define FAN_PWM_CHANNEL     LEDC_CHANNEL_0
#define FAN_PWM_MODE        LEDC_LOW_SPEED_MODE

void fan_init() {
    ledc_timer_config_t timer_config = {
        .speed_mode      = FAN_PWM_MODE,
        .timer_num       = FAN_PWM_TIMER,
        .duty_resolution = FAN_PWM_RESOLUTION,
        .freq_hz         = FAN_PWM_FREQ,
        .clk_cfg         = LEDC_AUTO_CLK
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer_config));

    ledc_channel_config_t channel_config = {
        .gpio_num   = FAN_PWM_GPIO,
        .speed_mode = FAN_PWM_MODE,
        .channel    = FAN_PWM_CHANNEL,
        .timer_sel  = FAN_PWM_TIMER,
        .duty       = 0,
        .hpoint     = 0
    };
    ESP_ERROR_CHECK(ledc_channel_config(&channel_config));
}

void fan_set_level(fan_level_t level) {
    uint16_t duty;

    switch(level) {
        case SPEED_OFF:
            duty = 0;          //0%
            break;

        case SPEED_LOW:
            duty = 256;        //25%
            break;

        case SPEED_MEDIUM:
            duty = 512;        //50%
            break;

        case SPEED_HIGH:
            duty = 768;        //75%
            break;

        case SPEED_MAX:
            duty = 1023;       //100%
            break;
            
        default:
            duty = 0;
            break;
    }

    ledc_set_duty(FAN_PWM_MODE, FAN_PWM_CHANNEL, duty);
    ledc_update_duty(FAN_PWM_MODE, FAN_PWM_CHANNEL);
}

void fan_stop(void) {
    uint8_t duty = 0;
    ledc_set_duty(FAN_PWM_MODE, FAN_PWM_CHANNEL, duty);
    ledc_update_duty(FAN_PWM_MODE, FAN_PWM_CHANNEL);
}
