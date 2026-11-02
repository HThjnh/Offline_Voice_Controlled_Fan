#include "audio.h"

#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"

static const char *TAG = "AUDIO";
static i2s_chan_handle_t rx_handle = NULL;

#define AUDIO_SAMPLE_RATE   16000
#define AUDIO_GPIO_BCLK     26
#define AUDIO_GPIO_WS       25
#define AUDIO_GPIO_DIN      33

#define AUDIO_BITS_PER_SAMPLE   I2S_DATA_BIT_WIDTH_32BIT
#define AUDIO_SLOT_MODE         I2S_SLOT_MODE_MONO

esp_err_t audio_init(void) {
    if (rx_handle != NULL) {
        ESP_LOGW(TAG, "I2S already initialized");
        return ESP_ERR_INVALID_STATE;
    }
    //Create channel I2S RX
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);

    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &rx_handle);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create I2S channel");
        rx_handle = NULL;
        return ret;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(AUDIO_SAMPLE_RATE),

        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(AUDIO_BITS_PER_SAMPLE, AUDIO_SLOT_MODE),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = AUDIO_GPIO_BCLK,
            .ws   = AUDIO_GPIO_WS,
            .dout = I2S_GPIO_UNUSED,
            .din  = AUDIO_GPIO_DIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ret = i2s_channel_init_std_mode(rx_handle, &std_cfg);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize I2S: %s", esp_err_to_name(ret));
        i2s_del_channel(rx_handle);
        rx_handle = NULL;
        return ret;
    }

    ret = i2s_channel_enable(rx_handle);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to enable I2S: %s", esp_err_to_name(ret));

        i2s_del_channel(rx_handle);
        rx_handle = NULL;

        return ret;
    }

    ESP_LOGI(TAG, "I2S initialized successfully");
    return ESP_OK;
}

esp_err_t audio_read(int32_t *buffer, size_t buffer_size, size_t *bytes_read) {
    //Check argument
    if (buffer == NULL || bytes_read == NULL || buffer_size == 0) {
        return ESP_ERR_INVALID_ARG;
    }
    //Check buffer size
    if (buffer_size % sizeof(int32_t) != 0) {
        return ESP_ERR_INVALID_SIZE;
    }
    //Check initialization
    if (rx_handle == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    *bytes_read = 0;

    return i2s_channel_read(rx_handle, buffer, buffer_size, bytes_read, pdMS_TO_TICKS(1000));
}
