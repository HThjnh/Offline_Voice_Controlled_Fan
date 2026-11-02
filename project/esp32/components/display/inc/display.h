#ifndef DISPLAY_H
#define DISPLAY_H
#include <stdbool.h>

/**
 * @brief Initialize the SSD1306 OLED display over the I2C bus.
 *
 * Configures GPIO21 as SDA and GPIO22 as SCL, creates the LCD panel,
 * initializes the framebuffer, and transfers its initial contents.
 *
 * @note This function uses ESP_ERROR_CHECK() and aborts on driver errors.
 */
void display_init(void);

/**
 * @brief Send the current framebuffer to the OLED display.
 *
 * Drawing functions only modify the framebuffer. Call this function
 * to make their changes visible on the physical screen.
 */

void display_update(void);

/**
 * @brief Clear the framebuffer and immediately refresh the display.
 */
void display_clear(void);

/**
 * @brief Set or clear one pixel in the framebuffer.
 *
 * Coordinates outside the 128x64 display are ignored.
 *
 * @param x Horizontal coordinate (0 to 127).
 * @param y Vertical coordinate (0 to 63).
 * @param color True to set the pixel, false to clear it.
 */
void display_draw_pixel(int x, int y, bool color);

/**
 * @brief Draw one 8x8 character in the framebuffer.
 *
 * The entire character must fit within the display area.
 *
 * @param x Horizontal coordinate of the character's top-left corner.
 * @param y Vertical coordinate of the character's top-left corner.
 * @param c Character to render using the font8x8 table.
 * @param color True for bright text on a dark background; false for inverse.
 */
void display_draw_char(int x, int y, char c, bool color);

/**
 * @brief Draw an 8-pixel-high string centered horizontally.
 *
 * The caller must supply a non-null, null-terminated string. For the
 * current implementation, use a non-empty string of at most 16 characters.
 * Call display_update() after drawing to refresh the OLED.
 *
 * @param y Vertical coordinate of the string's top edge.
 * @param text Null-terminated string to render.
 * @param color True for bright text; false for inverse text.
 */
void display_draw_string(int y, const char *text, bool color);

#endif