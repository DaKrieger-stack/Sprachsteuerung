idf_component_register(
    SRCS "main.cpp" "modell.c"
    INCLUDE_DIRS "." "../include"
    REQUIRES driver esp-dsp esp-tflite-micro
)
