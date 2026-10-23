#ifndef AUDIO_H
#define AUDIO_H

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

esp_err_t audio_init(void);
esp_err_t audio_read(int32_t *buffer, size_t buffer_size, size_t *bytes_read);

#endif 