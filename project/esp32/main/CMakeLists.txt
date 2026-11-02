idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES
        fan
        display
        preprocess
        command
        wav_reader
        encoder
        pid_controller
        audio
    EMBED_FILES
        "low.wav" "noise.wav" "unknown.wav" "high.wav" "off.wav"
)