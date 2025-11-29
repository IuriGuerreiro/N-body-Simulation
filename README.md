# N-Body Simulation (CUDA + OpenGL)

A real-time 3D gravitational N-body simulation leveraging NVIDIA CUDA for GPU-accelerated physics computation and OpenGL for interactive visualization.

## Overview

This project simulates the gravitational interactions between thousands of celestial bodies in 3D space. The simulation uses CUDA for parallel force calculations on the GPU and OpenGL for rendering, achieving real-time performance with up to 10,000+ bodies.

### Key Features

- **GPU-Accelerated Physics**: CUDA kernels compute gravitational forces in parallel across thousands of bodies
- **Real-Time Visualization**: OpenGL renders the simulation with interactive camera controls
- **CUDA-OpenGL Interop**: Direct GPU memory sharing eliminates CPU-GPU data transfer bottlenecks
- **3D Camera System**: Navigate through the simulation with keyboard controls (WASD + Q/E)
- **Configurable Parameters**: Easily adjust body count, gravitational constant, time step, and initial conditions

## Technical Architecture

### Physics Engine
- **Gravitational Force Calculation**: N² force computation with softening factor to prevent singularities
- **Numerical Integration**: Simple Euler method for position/velocity updates
- **Time Step**: `dt = 0.001` for stability
- **Gravitational Constant**: `G = 6.6743e-5` (scaled for simulation units)

### GPU Implementation
- **CUDA Kernels**: Three-stage simulation pipeline per frame
  1. `Reset_forces()`: Clear force accumulation arrays
  2. `Calculate_forces()`: Compute pairwise gravitational interactions using atomic operations
  3. `Calculate_pos()`: Update velocities and positions via numerical integration
- **Thread Organization**: 256 threads per block, dynamically calculated grid size
- **Memory Management**: CUDA-OpenGL interop with mapped buffer resources

### Rendering System
- **OpenGL 4.5 Core Profile**: Modern shader-based rendering
- **Vertex Buffer Object (VBO)**: Shared memory between CUDA and OpenGL
- **Point Rendering**: Each body rendered as a 5px point primitive
- **Projection System**: Custom scaling and view matrices for camera control

## Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability 7.5+ (Turing architecture or newer)
- Recommended: GTX 1660 or better for smooth performance with 10,000 bodies

### Software
- **CUDA Toolkit** (tested with CUDA 11+)
- **GLFW 3.x**: Window and input management
- **GLEW**: OpenGL extension loading
- **Windows OS**: Build script currently supports Windows with MSVC compiler
- **Visual Studio Build Tools**: C++ compiler with `/MD` and `/EHsc` flags

## Build Instructions

### Windows (MSVC)

1. **Ensure CUDA toolkit is installed** and `nvcc` is in your PATH
2. **Place required libraries** in the project structure:
   ```
   N-body-simulation/
   ├── include/
   │   ├── GL/          (GLEW headers)
   │   └── GLFW/        (GLFW headers)
   ├── lib/
   │   ├── glfw3.lib
   │   ├── glew32.lib
   │   └── (OpenGL libs provided by Windows SDK)
   ├── main.cu
   └── build.bat
   ```
3. **Run the build script**:
   ```cmd
   build.bat
   ```
4. **Execute the simulation**:
   ```cmd
   main.exe
   ```

### Build Configuration

The `build.bat` script compiles with:
- **Architecture**: `sm_75` (Turing GPUs - adjust for your GPU)
- **Compiler Flags**: `/MD` (dynamic runtime), `/EHsc` (C++ exception handling)
- **Libraries**: GLFW3, GLEW32, OpenGL32, GDI32, User32, Shell32

**Note**: Adjust `-arch=sm_75` to match your GPU's compute capability:
- RTX 20xx/GTX 16xx: `sm_75`
- RTX 30xx: `sm_86`
- RTX 40xx: `sm_89`

## Controls

| Key | Action |
|-----|--------|
| **W** | Move camera closer (zoom in) |
| **S** | Move camera farther (zoom out) |
| **A** | Pan camera left |
| **D** | Pan camera right |
| **Q** | Pan camera up |
| **E** | Pan camera down |

## Configuration Parameters

Edit `main.cu` to customize simulation parameters:

```cpp
// Simulation parameters
const int bodies_count = 10000;        // Number of bodies (adjust for performance)
const int seed = 12345;                // RNG seed for reproducibility
__device__ const float G = 6.6743e-5f; // Gravitational constant
__device__ const float dt = 0.001f;    // Time step (smaller = more accurate, slower)

// Initial conditions (in Initialize_bodies function)
std::uniform_real_distribution<float> mass_dist(1.0e5, 1.0e10);  // Body masses
std::uniform_real_distribution<float> pos_dist(-5.0e2, 5.0e2);   // Position range
std::uniform_real_distribution<float> vel_dist(-1.0e1, 1.0e1);   // Initial velocities

// Rendering
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
float softening_factor = 10.0f;        // Prevents extreme forces at close range
gl_PointSize = 5.0;                    // Point size in pixels (in vertex shader)
```

## Performance Characteristics

### Computational Complexity
- **O(N²) per frame**: Each body calculates forces from all other bodies
- **10,000 bodies**: ~100 million force calculations per frame
- **Typical Performance**: 30-60 FPS with modern NVIDIA GPUs

### Optimization Strategies
- **Atomic Operations**: Thread-safe force accumulation without race conditions
- **Softening Factor**: Prevents numerical instability when bodies are close
- **Frame Limiting**: `glfwSwapInterval(2)` caps at 30 FPS for smoother rendering
- **Minimal CPU-GPU Transfer**: Only debug readback every 60 frames

### Known Limitations
- N² algorithm doesn't scale to millions of bodies (consider Barnes-Hut or FMM for large-scale)
- Simple Euler integration can accumulate numerical errors over time
- No collision detection or merging of bodies

## Project Structure

```
N-body-simulation/
├── main.cu              # Main simulation and rendering code
├── build.bat            # Windows build script
├── include/             # Header files (GLFW, GLEW)
│   ├── GL/
│   └── GLFW/
├── lib/                 # Library files (.lib)
└── README.md            # This file
```

## Future Enhancements

- [ ] **Barnes-Hut Algorithm**: O(N log N) complexity for larger simulations
- [ ] **Leapfrog Integration**: More stable numerical integration
- [ ] **Collision Detection**: Handle body mergers and elastic collisions
- [ ] **Color Mapping**: Visualize velocity, mass, or energy
- [ ] **Configuration File**: Load initial conditions from JSON/YAML
- [ ] **Linux/Mac Support**: CMake build system for cross-platform compatibility
- [ ] **GPU Metrics**: Display FPS, body count, and performance stats
- [ ] **Presets**: Galaxy formation, solar system, binary stars, etc.

## License

This project is open-source. Feel free to modify and distribute.

## Acknowledgments

- **CUDA Programming**: NVIDIA CUDA Toolkit documentation
- **OpenGL Rendering**: Learn OpenGL tutorials
- **Physics**: Newtonian gravitation and N-body problem literature

---

**Author**: [Your Name]
**Date**: 2025-11-29
**GPU**: NVIDIA CUDA Compute Capability 7.5+
