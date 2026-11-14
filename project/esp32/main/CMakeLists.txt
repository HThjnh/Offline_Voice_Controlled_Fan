idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES
        fan
        display
        preprocess
        command
        encoder
        pid_controller
        audio
        control
)