batch@echo off
nvcc -o main main.cu ^
    -I"./include" ^
    -L"./lib" ^
    glfw3.lib glew32.lib opengl32.lib gdi32.lib user32.lib shell32.lib ^
    -arch=sm_75 ^
    --compiler-options "/MD /EHsc /NODEFAULTLIB:LIBCMT"
pause