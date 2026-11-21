#include "display.h"
#include "font.h"

#include "driver/i2c_master.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ssd1306.h"
#include "esp_lcd_panel_ops.h"

#include "esp_log.h"
#include <string.h>
#include <stdint.h>

#define DISPLAY_SDA     21
#define DISPLAY_SCL     22
#define DISPLAY_WIDTH   128
#define DISPLAY_HEIGHT  64

static i2c_master_bus_handle_t display_i2c_bus;
static esp_lcd_panel_io_handle_t display_io;
static esp_lcd_panel_handle_t display_panel;
static uint8_t display_buffer[DISPLAY_WIDTH * DISPLAY_HEIGHT / 8];

/**
 * @brief Configure and initialize the I2C SSD1306 display.
 *
 * Creates the I2C master bus, panel I/O interface, and SSD1306 panel.
 * Then clears the framebuffer and sends its contents to the screen.
 */
void display_init(void) {
    i2c_master_bus_config_t bus_config = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = DISPLAY_SDA,
        .scl_io_num = DISPLAY_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_config, &display_i2c_bus));
    // Check whether the SSD1306 acknowledges its I2C address.
    esp_err_t probe_ret = i2c_master_probe(
        display_i2c_bus,
        0x3C,
        1000
    );
    ESP_LOGI("OLED", "I2C probe: %s", esp_err_to_name(probe_ret));

    esp_lcd_panel_io_i2c_config_t io_config = {
        .dev_addr = 0x3C,
        .scl_speed_hz = 100000,
        .control_phase_bytes = 1,
        .dc_bit_offset = 6,
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_i2c(display_i2c_bus, &io_config, &display_io));

    esp_lcd_panel_dev_config_t panel_config = {
        .bits_per_pixel = 1,
        .reset_gpio_num = -1,
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_ssd1306(display_io, &panel_config, &display_panel));
    ESP_ERROR_CHECK(esp_lcd_panel_reset(display_panel));
    ESP_ERROR_CHECK(esp_lcd_panel_init(display_panel));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(display_panel, true, true));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(display_panel, true));

    // Clear RAM and update the panel once at startup.
    memset(display_buffer, 0x00, sizeof(display_buffer));
    display_update();
    ESP_LOGI("OLED", "Clear framebuffer sent");
}

/**
 * @brief Transfer the entire framebuffer to the SSD1306 display.
 */
void display_update(void)
{
    // Select Page Addressing Mode.
    uint8_t mode = 0x02;
    ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(display_io, 0x20, &mode, 1));

    // Write all 8 display pages.
    for (int page = 0; page < DISPLAY_HEIGHT / 8; page++) {
        // Select page 0 to 7.
        ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(display_io, 0xB0 + page, NULL, 0));
        // Set column address to 0.
        ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(display_io, 0x00, NULL, 0));
        ESP_ERROR_CHECK(esp_lcd_panel_io_tx_param(display_io, 0x10, NULL, 0));
        // Transfer 128 bytes for the selected page.
        ESP_ERROR_CHECK(esp_lcd_panel_io_tx_color(display_io, -1, &display_buffer[page * DISPLAY_WIDTH], DISPLAY_WIDTH));
    }
}


/**
 * @brief Clear all pixels and immediately refresh the OLED.
 */
void display_clear(void) {
    memset(display_buffer, 0x00, sizeof(display_buffer));
    display_update();
}

/**
 * @brief Set or clear a pixel in the local framebuffer.
 *
 * SSD1306 memory is organized into 8-pixel-high pages. Each byte stores
 * the vertical pixels at one horizontal coordinate within a page.
 */
void display_draw_pixel(int x, int y, bool color) {
    if (x < 0 || x > DISPLAY_WIDTH - 1 || y < 0 || y > DISPLAY_HEIGHT - 1) return;

    uint8_t page = y/8;
    uint16_t byte_index = page*128 + x;
    uint8_t bit = y%8;
    if(color) display_buffer[byte_index] |=  (1 << bit);
    else      display_buffer[byte_index] &= ~(1 << bit);
}

/**
 * @brief Draw an 8x8 glyph from the font table into the framebuffer.
 *
 * Uses the requested color for foreground pixels and the opposite
 * color for background pixels.
 */
void display_draw_char(int x, int y, char c, bool color) {
    if (x < 0 || x + 8 > DISPLAY_WIDTH || y < 0 || y + 8 > DISPLAY_HEIGHT) {
        return;
    }

    const uint8_t *bitmap = font8x8[(uint8_t)c];
    for (int row = 0; row < 8; row++) {
        uint8_t data = bitmap[row];
        for (int col = 0; col < 8; col++) {
            bool pixel = (data >> (7 - col)) & 1;
            display_draw_pixel(x + col, y + row, pixel ? color : !color);
        }
    }
}

/**
 * @brief Draw a horizontally centered string using 8x8 characters.
 *
 * This function updates the framebuffer only. Call display_update()
 * afterward to display the new text.
 *
 * @note The existing implementation expects a non-empty string.
 */
void display_draw_string(int y, const char *text, bool color) {
    int length = 0;
    while (text[length] != '\0') {
        length++;
    }

    int text_width = length * 8;
    int x = (DISPLAY_WIDTH  - text_width) / 2;
    for (int i = 0; i < length; i++) {
        display_draw_char(x + i * 8, y, text[i], color);
    }
}
