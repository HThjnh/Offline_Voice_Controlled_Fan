idf_component_register(
    SRCS "command.cpp" "kws_model_data.cc"
    INCLUDE_DIRS "inc"
    REQUIRES esp-tflite-micro
    PRIV_REQUIRES esp_timer
)