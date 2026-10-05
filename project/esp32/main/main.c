
#include "fan.h"
#include "display.h"
void app_main(void) {
    display_init();
    display_clear();

    display_draw_string(8, "SMART FAN", true);
    display_draw_string(24, "STATUS: ON", true);
    display_draw_string(40, "SPEED: 75%", true);

    display_update();
}