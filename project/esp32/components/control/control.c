#include "control.h"
/*
static fan_level_t current_level = SPEED_OFF;

void control_init(void) {
    fan_init();
    display_init();

    display_clear();
    display_draw_string(8,  "SMART FAN", true);
    display_draw_string(24, "STATUS: OFF", true);
    display_draw_string(40, "SPEED: 0%", true);
    display_draw_string(64, "LISTENING...", true);
    display_update();
}

void control_level(fan_level_t level) {
    fan_set_level(level);

    display_clear();
    display_draw_string(8, "SMART FAN", true);

    switch (level) {
        case SPEED_OFF: 
            display_draw_string(24, "STATUS: OFF", true);
            display_draw_string(40, "SPEED: 0%", true);
            display_draw_string(64, "RECOGNIZED")
            break;

        case SPEED_LOW:
            display_draw_string(24, "STATUS: ON", true);
            display_draw_string(40, "SPEED: 25%", true);
            break;

        case SPEED_MEDIUM:
            display_draw_string(24, "STATUS: ON", true);
            display_draw_string(40, "SPEED: 50%", true);
            break;

        case SPEED_HIGH:
            display_draw_string(24, "STATUS: ON", true);
            display_draw_string(40, "SPEED: 75%", true);
            break;

        case SPEED_MAX:
            display_draw_string(24, "STATUS: ON", true);
            display_draw_string(40, "SPEED: 100%", true);
            break;

        default:
            return;
    }

}
*/


