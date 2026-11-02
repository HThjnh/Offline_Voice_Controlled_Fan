idf_component_register(
    SRCS "preprocess.c"
    INCLUDE_DIRS "inc"
    REQUIRES command
    PRIV_REQUIRES log esp-dsp
)