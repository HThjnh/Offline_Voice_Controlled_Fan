#ifndef DISPLAY_H
#define DISPLAY_H
#include <stdbool.h>

void display_init(void);
void display_update(void);
void display_clear(void);
void display_draw_pixel(int x, int y, bool color);
void display_draw_char(int x, int y, char c, bool color);
void display_draw_string(int y, const char *text, bool color);

#endif