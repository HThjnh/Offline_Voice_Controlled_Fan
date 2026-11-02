#include "display.h"
#include "font.h"
#include "driver/i2c_master.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_ssd1306.h"
#include "esp_lcd_panel_ops.h"
#include <string.h>

#define DISPLAY_SDA     21
#define DISPLAY_SCL     22

static i2c_master_bus_handle_t display_i2c_bus;
static esp_lcd_panel_io_handle_t display_io;
static esp_lcd_panel_handle_t display_panel;
static uint8_t display_buffer[128 * 64 / 8];

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

    esp_lcd_panel_io_i2c_config_t io_config = {
        .dev_addr = 0x3C,
        .scl_speed_hz = 400000,
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
    memset(display_buffer, 0x00, sizeof(display_buffer));

    display_update();
}

/**
 * @brief Transfer the entire framebuffer to the SSD1306 display.
 */
void display_update(void) {
    ESP_ERROR_CHECK(esp_lcd_panel_draw_bitmap(display_panel,0,0,128,64,display_buffer));
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
    if (x < 0 || x > 127 || y < 0 || y > 63) return;

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
    if (x < 0 || x + 8 > 128 || y < 0 || y + 8 > 64) {
        return;
    }

    const uint8_t *bitmap = font8x8[(uint8_t)c];

    for (int row = 0; row < 8; row++) {

        uint8_t data = bitmap[row];

        for (int col = 0; col < 8; col++) {

            bool pixel = (data >> (7 - col)) & 1;

            display_draw_pixel(
                x + col,
                y + row,
                pixel ? color : !color
            );
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
    int x = (128 - text_width) / 2;

    display_draw_char(x, y, text[0], color);

    for (int i = 0; i < length; i++) {
        display_draw_char(x + i * 8, y, text[i], color);
    }
}

