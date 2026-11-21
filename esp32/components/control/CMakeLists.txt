idf_component_register(
    SRCS "control.c"
    INCLUDE_DIRS "inc"
    REQUIRES fan encoder pid_controller
)