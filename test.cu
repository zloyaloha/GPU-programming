#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count;
    cudaGetDeviceCount(&count);
    std::cout << "Devices: " << count << "\n";
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    std::cout << "Device 0: " << p.name << " SM " << p.major << "." << p.minor << "\n";
}