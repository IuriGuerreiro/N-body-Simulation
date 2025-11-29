#include "include/GL/glew.h"
#include "include/GLFW/glfw3.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm> // For std::min

// CUDA Headers
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
__device__ const float G = 6.6743e-5f;
__device__ const float dt = 0.001f; // time step
const int bodies_count = 10000;
const int seed = 12345;


// Camera position (you can modify these with keyboard)
float camera_x = 0.0f;
float camera_y = 0.0f;
float camera_z = 2000.0f; // Adjusted for better initial visibility (was 5000.0f)
float camera_zoom = 1.0f;

struct body{
    float mass;
    float position[3]; // 3D position (offset: 4 bytes)
    float velocity[3]; // 3D position (offset: 16 bytes)
    float force[3]; // 3D position (offset: 28 bytes)
    // Total size: 4 + 12 + 12 + 12 = 40 bytes. OpenGL only cares about the position offset.

    body() {}

    body(float m, float pos[3], float vel[3]){
        mass = m;
        position[0] = pos[0];
        position[1] = pos[1];
        position[2] = pos[2];
        velocity[0] = vel[0];
        velocity[1] = vel[1];
        velocity[2] = vel[2];
        force[0] = 0.0f;
        force[1] = 0.0f;
        force[2] = 0.0f;
    }
};

void Initialize_bodies(body* bodies){
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> mass_dist(1.0e5, 1.0e10);
    std::uniform_real_distribution<float> pos_dist(-5.0e2, 5.0e2);
    std::uniform_real_distribution<float> vel_dist(-1.0e1, 1.0e1);

    for(int i = 0; i < bodies_count; i++){
        float mass = mass_dist(rng);

        float position[3] = {pos_dist(rng), pos_dist(rng), pos_dist(rng)};
        float velocity[3] = {vel_dist(rng), vel_dist(rng), vel_dist(rng)};
        
        bodies[i] = body{mass, position, velocity};
    }
}


__device__ void Reset_forces(body* bodies,  int num_bodies){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_bodies) {
        bodies[i].force[0] = 0.0f;
        bodies[i].force[1] = 0.0f;
        bodies[i].force[2] = 0.0f;
    }
}

__device__ void Calculate_forces(body* bodies, int num_bodies){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bodies) return;

    for (int j = 0; j < num_bodies; j++) {
        if (i == j) continue;  // SKIP SELF-INTERACTION
        float r_vec[3] = {
            bodies[i].position[0] - bodies[j].position[0],
            bodies[i].position[1] - bodies[j].position[1],
            bodies[i].position[2] - bodies[j].position[2]
        };

        float r_scalar = sqrtf(
            r_vec[0] * r_vec[0] +
            r_vec[1] * r_vec[1] +
            r_vec[2] * r_vec[2]
        );

        // Softening factor (epsilon) to prevent extreme forces when bodies are too close
        float softening_factor = 10.0f; 
        float r_squared = r_scalar * r_scalar + softening_factor * softening_factor;
        float r_cubed_inv = 1.0f / (r_scalar * r_squared);

        // Magnitude of force F = G * m1 * m2 / r^2
        // We calculate F/r = G * m1 * m2 / r^3 for the vector calculation
        float magnitude_factor = G * bodies[i].mass * bodies[j].mass * r_cubed_inv;
        
        float force_vec[3] = {
            magnitude_factor * r_vec[0],
            magnitude_factor * r_vec[1],
            magnitude_factor * r_vec[2]
        };
        
        // Accumulate forces (attractive, so -force_vec)
        atomicAdd(&bodies[i].force[0], -force_vec[0]);
        atomicAdd(&bodies[i].force[1], -force_vec[1]);
        atomicAdd(&bodies[i].force[2], -force_vec[2]);
    }
}


__device__ void Calculate_pos(body* bodies, int num_bodies){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_bodies) return;

    if (i < num_bodies) {
        float accel[3] = {
            bodies[i].force[0] / bodies[i].mass,
            bodies[i].force[1] / bodies[i].mass,
            bodies[i].force[2] / bodies[i].mass
        };
        bodies[i].velocity[0] += accel[0] * dt;
        bodies[i].velocity[1] += accel[1] * dt;
        bodies[i].velocity[2] += accel[2] * dt; 
        bodies[i].position[0] += bodies[i].velocity[0] * dt;
        bodies[i].position[1] += bodies[i].velocity[1] * dt;
        bodies[i].position[2] += bodies[i].velocity[2] * dt;
    }
}

// simulate each step of the N-body problem on the GPU
__global__ void simulate_step(body* bodies, int num_bodies) {
    
    // Reset forces (each body's force reset by one thread)
    Reset_forces(bodies, num_bodies);
    __syncthreads();
    
    // Calculate new forces (forces accumulate via atomics)
    Calculate_forces(bodies, num_bodies);
    __syncthreads();
    
    // Update position and velocity
    Calculate_pos(bodies, num_bodies);
    __syncthreads();
}

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cout << "Shader compilation failed: " << infoLog << std::endl;
    }
    return shader;
}

GLuint createShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    int success;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cout << "Program linking failed: " << infoLog << std::endl;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

// Function to update view matrix based on camera position
void updateViewMatrix(float view[16]) {
    // Simple view matrix: move camera back
    view[0] = 1; view[1] = 0; view[2] = 0; view[3] = 0;
    view[4] = 0; view[5] = 1; view[6] = 0; view[7] = 0;
    view[8] = 0; view[9] = 0; view[10] = 1; view[11] = 0;
    view[12] = -camera_x;
    view[13] = -camera_y;
    view[14] = -camera_z;
    view[15] = 1;
}

// Function to handle keyboard input
void handleInput(GLFWwindow* window) {
    float moveSpeed = 10.0f;
    float zoomSpeed = 20.0f;
    
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        camera_z -= zoomSpeed;  // Move closer
        // std::cout << "Camera Z: " << camera_z << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        camera_z += zoomSpeed;  // Move farther
        // std::cout << "Camera Z: " << camera_z << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
        camera_x -= moveSpeed;  // Pan left
        // std::cout << "Camera X: " << camera_x << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
        camera_x += moveSpeed;  // Pan right
        // std::cout << "Camera X: " << camera_x << std::endl;
    }
    
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS){
        camera_y += moveSpeed;  // Pan up
        // std::cout << "Camera Y: " << camera_y << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS){
        camera_y -= moveSpeed;  // Pan down
        // std::cout << "Camera Y: " << camera_y << std::endl;
    }
    
    // Keep camera from going too close
    if (camera_z < 50.0f) // Allow getting closer
        camera_z = 50.0f;
}


void centerCameraOnBodies(void* dev_ptr) {
    // NOTE: This function requires a blocking copy (cudaMemcpy) to the CPU, 
    // which can significantly slow down the rendering loop. 
    // A better approach would be to calculate the center of mass on the GPU.
    // However, for debugging purposes, we will keep it simple.
    static int frameCount = 0;
    if (frameCount++ % 60 != 0) return;  // Only update every 60 frames
    
    body temp;
    float center_x = 0, center_y = 0, center_z = 0;
    
    // Sample only 100 bodies to reduce copy overhead
    int sample_size = std::min(100, bodies_count); 
    
    for (int i = 0; i < sample_size; i++) {
        cudaMemcpy(&temp, (body*)dev_ptr + i, sizeof(body), cudaMemcpyDeviceToHost);
        center_x += temp.position[0];
        center_y += temp.position[1];
        center_z += temp.position[2];
    }
    
    center_x /= sample_size;
    center_y /= sample_size;
    center_z /= sample_size;
    
    // Move camera to follow center
    camera_x = center_x;
    camera_y = center_y;
    camera_z = center_z + 2000.0f;  // Offset so we're looking at it
    
    // std::cout << "Center at: (" << center_x << ", " << center_y << ", " << center_z << ")" << std::endl;
}


int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Set OpenGL window hints for version 4.5 (or whatever you use)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, 
                                             "N-Body CUDA Interop", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(2); // make it 30 FPS cap

    // Initialize GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(err) << std::endl;
        return -1;
    }

    // ====================================================================
    // CRITICAL OPENGL STATE FIXES
    // ====================================================================

    // FIX 1: Allow gl_PointSize to be set in the vertex shader.
    glEnable(GL_PROGRAM_POINT_SIZE); 
    
    // FIX 2: Enable depth testing to correctly render 3D
    glEnable(GL_DEPTH_TEST);

    // Optional: Enable blending for anti-aliasing (smoother points)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // ====================================================================

    body* C_bodies = (body*)malloc(sizeof(body) * bodies_count);

    Initialize_bodies(C_bodies);
    std::cout << "Initialized " << bodies_count << " bodies." << std::endl;

    // Setup OpenGL VBO and buffer
    GLuint VBO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Allocate space for all bodies
    glBufferData(GL_ARRAY_BUFFER, sizeof(body) * bodies_count, C_bodies, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // NOW register with CUDA (AFTER allocation)
    cudaGraphicsResource *DV_bodies;

    cudaError_t error = cudaGraphicsGLRegisterBuffer(
        &DV_bodies,
        VBO,
        cudaGraphicsMapFlagsWriteDiscard 
    );

    if (error != cudaSuccess) {
        std::cerr << "CUDA VBO registration failed: " << cudaGetErrorString(error) << std::endl;
    }

    // Map it once before render loop
    cudaGraphicsMapResources(1, &DV_bodies, 0);

    size_t num_bytes;
    void *dev_ptr;
    cudaGraphicsResourceGetMappedPointer(&dev_ptr, &num_bytes, DV_bodies);

    // Copy initial data to GPU
    cudaMemcpy(dev_ptr, C_bodies, sizeof(body) * bodies_count, cudaMemcpyHostToDevice);

    // Setup OpenGL VAO
    GLuint VAO;
    glGenVertexArrays(1, &VAO);

    // activate VAO and VBO
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    // Attribute 0: position[3]
    glVertexAttribPointer(
        0, 3, GL_FLOAT, GL_FALSE,
        sizeof(body),
        (void*)offsetof(body, position)
    );
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);


    const char* vertexShader = R"(
    #version 330 core
    layout(location = 0) in vec3 position;

    uniform mat4 projection;
    uniform mat4 view;

    void main() {
        gl_Position = projection * view * vec4(position, 1.0);
        // gl_PointSize is now correctly enabled by GL_PROGRAM_POINT_SIZE on the host.
        // We set it large enough to be easily visible.
        gl_PointSize = 5.0; 
    }
    )";

    // FRAGMENT SHADER: Runs on GPU for EACH pixel being drawn
    const char* fragmentShader = R"(
    #version 330 core
    out vec4 FragColor;  // Output: final pixel color

    void main() {
        // Simple white color
        FragColor = vec4(1.0, 1.0, 1.0, 1.0); 
    }
    )";

    // Compile and link shaders
    GLuint shaderProgram = createShaderProgram(vertexShader, fragmentShader);

    float view[16]; // Will be updated every frame

    // PROJECTION MATRIX FIX:
    // The original matrix was not scaling the world coordinates (e.g., -500 to 500)
    // into the [-1, 1] clip space correctly. 
    // We use a simple scaling factor (0.005) to bring the cluster into view 
    // when the camera is around 2000-5000 units away.
    float scale_factor = 0.005f; 
    float projection[16] = {
        scale_factor, 0, 0, 0,
        0, scale_factor, 0, 0,
        0, 0, 0.001f, 0, 
        0, 0, 0, 1
    };

    // Send matrices to shader
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");

    glUseProgram(shaderProgram); // Activate shader before setting uniforms
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);

    while (!glfwWindowShouldClose(window)) {
        handleInput(window);

        // Update view matrix based on camera_x/y/z
        updateViewMatrix(view);
        
        static int frameCount = 0;
        if (frameCount++ % 60 == 0) {
            body temp;
            // Read body 0 data for debugging (can be commented out for performance)
            cudaMemcpy(&temp, (body*)dev_ptr, sizeof(body), cudaMemcpyDeviceToHost);
            std::cout << "Body 0 pos: " << temp.position[0] << ", " 
                    << temp.position[1] << ", " << temp.position[2] << std::endl;
            std::cout << "Camera at: (" << camera_x << ", " << camera_y << ", " << camera_z << ")" << std::endl;
        }
        
        // Send updated view matrix to shader
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, view);

        // Execute CUDA kernel
        int blockSize = 256;
        int gridSize = (bodies_count + blockSize - 1) / blockSize;
        simulate_step<<<gridSize, blockSize>>>((body*)dev_ptr, bodies_count);
        cudaDeviceSynchronize();

        // Then draw
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear depth buffer too
        
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        
        glDrawArrays(GL_POINTS, 0, bodies_count);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // Clean up
    cudaGraphicsUnmapResources(1, &DV_bodies, 0);
    cudaGraphicsUnregisterResource(DV_bodies);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
    glDeleteProgram(shaderProgram);
    
    free(C_bodies);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}