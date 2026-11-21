#include "fan.h"

#include <stdbool.h>
#include "driver/ledc.h"
#include "driver/gpio.h"
#include "esp_err.h"
#include "esp_log.h"

#define FAN_PWM_GPIO        18
#define FAN_AIN1_GPIO       16
#define FAN_AIN2_GPIO       17
#define FAN_STBY_GPIO       23

#define FAN_PWM_FREQ        20000
#define FAN_PWM_RESOLUTION  LEDC_TIMER_10_BIT
#define FAN_PWM_TIMER       LEDC_TIMER_0
#define FAN_PWM_CHANNEL     LEDC_CHANNEL_0
#define FAN_PWM_MODE        LEDC_LOW_SPEED_MODE
#define FAN_PWM_MAX         1023

// 0: Disable motor output for software testing
// 1: Enable real motor control after hardware checks
#define FAN_ENABLE_OUTPUT  1

static const char *TAG = "FAN";
static bool fan_initialized = false;

/**
 * @brief Initialize the TB6612FNG motor driver and LEDC PWM.
 *
 * Configures the motor direction pins, standby pin, PWM timer,
 * and PWM output channel.
 */
void fan_init() {
    // Configure direction and standby pins
    gpio_config_t gpio_cfg = {
        .pin_bit_mask = (1ULL << FAN_AIN1_GPIO) | (1ULL << FAN_AIN2_GPIO) | (1ULL << FAN_STBY_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    ESP_ERROR_CHECK(gpio_config(&gpio_cfg));

    // Keep driver disabled during initialization
    ESP_ERROR_CHECK(gpio_set_level(FAN_STBY_GPIO, 0));
    ESP_ERROR_CHECK(gpio_set_level(FAN_AIN1_GPIO, 0));
    ESP_ERROR_CHECK(gpio_set_level(FAN_AIN2_GPIO, 0));

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
    ESP_ERROR_CHECK(ledc_fade_func_install(0));
    fan_initialized = true;
    fan_set_pwm(0);

    ESP_LOGI(TAG, "TB6612FNG initialized");

#if FAN_ENABLE_OUTPUT == 0
    ESP_LOGW(TAG, "Motor output disabled (test mode)");
#endif
}

/**
 * @brief Set the motor speed using PWM duty cycle.
 *
 * A zero duty cycle disables the motor driver.
 * A non-zero duty cycle enables forward rotation.
 *
 * @param duty PWM duty cycle (0 to FAN_PWM_MAX).
 */
void fan_set_pwm(uint16_t duty) {
    if (!fan_initialized) {
        ESP_LOGE(TAG, "fan_init() not called");
        return;
    }
    if (duty > FAN_PWM_MAX) {
        duty = FAN_PWM_MAX;
    }
#if FAN_ENABLE_OUTPUT == 0
    duty = 0;
#endif
    if (duty == 0) {
        // Disable motor driver
        ESP_ERROR_CHECK(gpio_set_level(FAN_STBY_GPIO, 0));
        ESP_ERROR_CHECK(ledc_set_duty_and_update(FAN_PWM_MODE, FAN_PWM_CHANNEL, 0, 0));
        ESP_ERROR_CHECK(gpio_set_level(FAN_AIN1_GPIO, 0));
        ESP_ERROR_CHECK(gpio_set_level(FAN_AIN2_GPIO, 0));
        return;
    } 
    // Set forward rotation
    ESP_ERROR_CHECK(gpio_set_level(FAN_AIN1_GPIO, 1));
    ESP_ERROR_CHECK(gpio_set_level(FAN_AIN2_GPIO, 0));
    // Update PWM duty
    ESP_ERROR_CHECK(ledc_set_duty_and_update(FAN_PWM_MODE, FAN_PWM_CHANNEL, duty, 0));

    // Enable motor driver.
    ESP_ERROR_CHECK(gpio_set_level(FAN_STBY_GPIO, 1));

    // Debug only to avoid excessive serial logging.
    ESP_LOGD(TAG, "PWM duty = %u", (unsigned)duty);
}

