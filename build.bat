@echo off
nvcc -o main main.cu ^
  -I"F:\Projects\Programmes\Simulations\N-body-simulation\include\GL" ^
  -I"F:\Projects\Programmes\Simulations\N-body-simulation\include\GLFW" ^
  -L"F:\Projects\Programmes\Simulations\N-body-simulation\lib" ^
  -L"F:\Projects\Programmes\Simulations\N-body-simulation\lib" ^
  -lglfw3 -lglew32 -lopengl32 -lgdi32
pause