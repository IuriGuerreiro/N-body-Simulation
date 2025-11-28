#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>
#include <GL/glew.h>      // OpenGL Extension Wrangler - gives you GPU functions
#include <GLFW/glfw3.h>    // Creates window and handles input
#include <glm/glm.hpp>     // Math library (vectors, matrices)


__device__ const float G = 6.6743e-11f;
__device__ const float dt = 1.0f; // 1s
const int bodies_count = 10000;
__device__ const int D_bodies_count = bodies_count;
const int seed = 12345;

// body struct definition
struct body{
    float mass;
    float position[3]; // 3D position
    float velocity[3]; // 3D position
    float force[3]; // 3D position

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
    std::uniform_real_distribution<float> pos_dist(-1.0e3, 1.0e3);
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

    for (int j = i+1; j < num_bodies; j++) {
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

        if (r_scalar < 1e-6) { continue; }

        float magnitude_factor = G * bodies[i].mass * bodies[j].mass / (r_scalar * r_scalar * r_scalar);
        
        float force_vec[3] = {
            magnitude_factor * r_vec[0],
            magnitude_factor * r_vec[1],
            magnitude_factor * r_vec[2]
        };
        
        bodies[i].force[0] -= force_vec[0];
        bodies[i].force[1] -= force_vec[1];
        bodies[i].force[2] -= force_vec[2];

        bodies[j].force[0] += force_vec[0];
        bodies[j].force[1] += force_vec[1];
        bodies[j].force[2] += force_vec[2];
    }
}


__device__ void Calculate_pos(body* bodies, int num_bodies){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

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
    Reset_forces(bodies, num_bodies);
    __syncthreads();
    
    Calculate_forces(bodies, num_bodies);
    __syncthreads();
    
    Calculate_pos(bodies, num_bodies);
    __syncthreads();
}

int main() {
    
    body* C_bodies = (body*)malloc(sizeof(body) * bodies_count);
    body* D_bodies;

    Initialize_bodies(C_bodies);
    std::cout << "Initialized " << bodies_count << " bodies." << std::endl;

    cudaMalloc(&D_bodies, sizeof(body) * bodies_count);
    cudaMemcpy(D_bodies, C_bodies, sizeof(body) * bodies_count, cudaMemcpyHostToDevice);


    int blockSize = 256;
    int gridSize = (bodies_count + blockSize - 1) / blockSize;
    std::cout << "Running GPU simulation..." << std::endl;

    for (int step = 0; step < 100; step++) {
        simulate_step<<<gridSize, blockSize>>>(D_bodies, bodies_count);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(C_bodies, D_bodies, sizeof(body) * bodies_count, cudaMemcpyDeviceToHost);

    std::cout << "Final position body 0: (" 
              << C_bodies[0].position[0] << ", " 
              << C_bodies[0].position[1] << ", " 
              << C_bodies[0].position[2] << ")" << std::endl;

    cudaFree(D_bodies);
    free(C_bodies);
    return 0;
}