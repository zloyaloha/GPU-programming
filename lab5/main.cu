#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>
#include <cassert>

const int MAX_SHARED_ELEMENTS = 1024;

inline void checkCuda(cudaError_t err, const char *msg = nullptr) {
    if (err != cudaSuccess) {
        if (msg) std::cerr << msg << ": ";
        std::cerr << cudaGetErrorString(err) << " (" << err << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

__global__ void bitonic_global_merge(int *data, unsigned int m, unsigned int distance, unsigned int block_size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; // айди нити
    if (tid >= m) return;
    unsigned int ix = tid ^ distance;

    if (ix > tid && ix < m) {
        bool dir = ((tid & block_size) == 0); // если тред айди справа в блоке, то по убыванию, иначе по возрастанию
        int a = data[tid];
        int b = data[ix];
        if ((a > b) == dir) {
            data[tid] = b;
            data[ix] = a;
        }
    }
}

void bitonic_cpu_merge(int *data, unsigned int m, unsigned int distance, unsigned int block_size) {
    for (int tid = 0; tid < m; ++tid) {
        if (tid >= m) return;
        unsigned int ix = tid ^ distance;
        if (ix > tid && ix < m) {
            bool dir = ((tid & block_size) == 0);
            int a = data[tid];
            int b = data[ix];
            if ((a > b) == dir) {
                data[tid] = b;
                data[ix] = a;
            }
        }
    }
}

__global__ void bitonic_block_sort_shared(int *data, unsigned int m, unsigned int seg_len) {
    extern __shared__ int s_data[];
    unsigned int tid = threadIdx.x;
    unsigned int block_start = blockIdx.x * seg_len;
    unsigned int global_idx = block_start + tid;

    if (tid < seg_len) {
        s_data[tid] = (global_idx < m) ? data[global_idx] : INT_MAX;
    }
    __syncthreads();


    for (unsigned int block_size = 2; block_size <= seg_len; block_size <<= 1) {
        for (unsigned int distance = block_size >> 1; distance > 0; distance >>= 1) {
            unsigned int ix = tid ^ distance;
            if (ix > tid && ix < seg_len) {
                bool dir = ((tid & block_size) == 0);
                int a = s_data[tid];
                int b = s_data[ix];
                if ((a > b) == dir) {
                    s_data[tid] = b;
                    s_data[ix] = a;
                }
            }
            __syncthreads();
        }
    }

    if (tid < seg_len && global_idx < m) {
        data[global_idx] = s_data[tid];
    }
}

unsigned int next_pow_2(unsigned int v) {
    if (v == 0) return 1;
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
}

void bitonic_sort_adaptive(int *d_data, unsigned int n) {
    unsigned int m = next_pow_2(n);

    double sum = 0;
    int kernel_n = 0;

    if (m <= MAX_SHARED_ELEMENTS) {
        std::cout << "USING SHARED MEMORY\n";
        unsigned int seg_len = 1024;
        unsigned int seg_blocks = (m + seg_len - 1) / seg_len;
        unsigned int th = seg_len;
        if (th > 1024u) th = 1024u;
        size_t shmem = seg_len * sizeof(int);
        cudaEvent_t start, stop;
        checkCuda(cudaEventCreate(&start));
        checkCuda(cudaEventCreate(&stop));
        checkCuda(cudaEventRecord(start));
        bitonic_block_sort_shared<<<1, th, shmem>>>(d_data, m, seg_len);
        checkCuda(cudaGetLastError(), "block_sort_shared launch failed");
        checkCuda(cudaEventRecord(stop));
        checkCuda(cudaEventSynchronize(stop));
        checkCuda(cudaGetLastError());
        float t;
        checkCuda(cudaEventElapsedTime(&t, start, stop));
        checkCuda(cudaEventDestroy(start));
        checkCuda(cudaEventDestroy(stop));
        ++kernel_n;
        sum += t;
    } else {
        std::cout << "USING GLOBAL MEMORY\n";
        for (unsigned int block_size = 2; block_size <= m; block_size <<= 1) {
            for (unsigned int distance = block_size >> 1; distance > 0; distance >>= 1) {
                cudaEvent_t start, stop;
                checkCuda(cudaEventCreate(&start));
                checkCuda(cudaEventCreate(&stop));
                checkCuda(cudaEventRecord(start));
                bitonic_global_merge<<<8192, 768>>>(d_data, m, distance, block_size);
                checkCuda(cudaGetLastError(), "global_merge launch failed");
                checkCuda(cudaEventRecord(stop));
                checkCuda(cudaEventSynchronize(stop));
                checkCuda(cudaGetLastError());
                float t;
                checkCuda(cudaEventElapsedTime(&t, start, stop));
                checkCuda(cudaEventDestroy(start));
                checkCuda(cudaEventDestroy(stop));
                ++kernel_n;
                sum += t;
            }
        }
    }
    std::cout << "Total: " << sum << " Average: " << sum / kernel_n << std::endl;
}

std::vector<int> read_to_vec() {
    int n;
    if (!(std::cin >> n)) {
        return std::vector<int>();
    }
    std::vector<int> nums(n);
    for (int i = 0; i < n; ++i) std::cin >> nums[i];
    return nums;
}

void write_to_cout(const std::vector<int>& nums) {
    for (size_t i = 0; i < nums.size(); ++i) {
        std::cout << nums[i] << (i + 1 == nums.size() ? '\n' : ' ');
    }
}

std::vector<int> read_to_vec_bin()
{
    int n;
    std::cin.read(reinterpret_cast<char*>(&n), sizeof(int));
    std::vector<int> nums(n); std::cin.read(reinterpret_cast<char*>(nums.data()), n * sizeof(int));
    return nums;
}
void write_to_cout_bin(std::vector<int>& nums)
{
    std::cout.write(reinterpret_cast<char*>(nums.data()), nums.size() * sizeof(int));
}

int main() {
    std::vector<int> h_data = read_to_vec();
    unsigned int n = static_cast<unsigned int>(h_data.size());
    if (n == 0) return 0;

    unsigned int m = next_pow_2(n);
    std::vector<int> h_pad(m, INT_MAX);
    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < n; ++i) h_pad[i] = h_data[i];
    for (unsigned int block_size = 2; block_size <= m; block_size <<= 1) {
        for (unsigned int distance = block_size >> 1; distance > 0; distance >>= 1) {
            // bitonic_cpu_merge(h_pad.data(), m, distance, block_size);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Время выполнения: " << diff.count() * 1000 << " миллисекунд" << std::endl;

    int *d_data = nullptr;
    checkCuda(cudaMalloc(&d_data, m * sizeof(int)), "cudaMalloc failed");
    checkCuda(cudaMemcpy(d_data, h_pad.data(), m * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H2D failed");

    bitonic_sort_adaptive(d_data, n);

    checkCuda(cudaMemcpy(h_pad.data(), d_data, m * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H failed");
    checkCuda(cudaFree(d_data), "cudaFree failed");

    std::vector<int> result(n);
    for (unsigned int i = 0; i < n; ++i) result[i] = h_pad[i];
    // write_to_cout(result);
    return 0;
}