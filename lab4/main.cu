#include <iostream>
#include <cuda_runtime.h>
#include <thrust/swap.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define CSC(call)  									                \
do {											                    \
	cudaError_t res = call;							                \
	if (res != cudaSuccess) {							            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);								                    \
	}										                        \
} while(0)

struct AbsComparator
{
    __host__ __device__ bool operator()(double a, double b)
    {
        return fabs(a) < fabs(b);
    }
};

__device__ int ij_to_lin(int row, int col, int n)
{
    return n * col + row;
}

__global__ void swap_rows(double *mat, int n, int row_a, int row_b)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int col = idx; col < n; col += offset) {
        double val_a = mat[ij_to_lin(row_a, col, n)];
        double val_b = mat[ij_to_lin(row_b, col, n)];
        mat[ij_to_lin(row_a, col, n)] = val_b;
        mat[ij_to_lin(row_b, col, n)] = val_a;
    }
}

__global__ void divide(double *mat, int n, int diag_i)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    for (int row = idx + diag_i + 1; row < n; row += offset) {
        mat[ij_to_lin(row, diag_i, n)] /= mat[ij_to_lin(diag_i, diag_i, n)];
    }
}

__global__ void kernel(double *data, int n, int diag_i)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int col = idy + diag_i + 1; col < n; col += offsety) {
        double pivot_row_val = data[ij_to_lin(diag_i, col, n)];
        for (int row = idx + diag_i + 1; row < n; row += offsetx) {
            data[ij_to_lin(row, col, n)] -= pivot_row_val * data[ij_to_lin(row, diag_i, n)];
        }
    }
}

struct ThrustInitializer {
    ThrustInitializer() {
        cudaError_t set_device_err = cudaSetDevice(0);
        if (set_device_err != cudaSuccess) {
            std::cerr << "FATAL THRUST INITIALIZATION FAILED BEFORE MAIN: "
                      << cudaGetErrorString(set_device_err)
                      << " (" << set_device_err << ")"
                      << std::endl;
            // Внимание: Здесь мы не можем использовать return 1, но можем вызвать exit.
            exit(1);
        }
        // std::cerr << "DEBUG: CUDA successfully initialized by ThrustInitializer." << std::endl;
    }
};

// Глобальный статический объект, который запускает инициализацию до main()
ThrustInitializer global_thrust_initializer;

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    int n;
    std::cin >> n;

    int final_size = n * n;

    double *mat = new double[final_size];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> mat[i + j * n];
        }
    }

    double *d_mat;
    CSC(cudaMalloc((void **) &d_mat, sizeof(double) * (final_size)));
    CSC(cudaMemcpy(d_mat, mat, sizeof(double) * (final_size), cudaMemcpyHostToDevice));

    int num_swaps = 0;
    AbsComparator comparator;
    thrust::device_ptr<double> i_col_ptr, i_max_elem_in_col_ptr;
    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));
    for (int i = 0; i < n - 1; ++i) {
        int idx_max_in_col;
        i_col_ptr = thrust::device_pointer_cast(d_mat + i * n);
        i_max_elem_in_col_ptr = thrust::max_element(i_col_ptr + i, i_col_ptr + n, comparator);
        idx_max_in_col = i_max_elem_in_col_ptr - i_col_ptr;


        double pivot_value;
        CSC(cudaMemcpy(&pivot_value,
                       thrust::raw_pointer_cast(i_max_elem_in_col_ptr),
                       sizeof(double),
                       cudaMemcpyDeviceToHost));

        if (fabs(pivot_value) < 1e-7) {
            std::cout << 0;
            break;
        }

        if (i != idx_max_in_col) {
            swap_rows<<<1024, 1024>>>(d_mat, n, i, idx_max_in_col);
            num_swaps++;
            CSC(cudaDeviceSynchronize());
        }
        divide<<<1024, 1024>>>(d_mat, n, i);
        CSC(cudaDeviceSynchronize());
        kernel<<<dim3(32, 32), dim3(32, 32)>>>(d_mat, n, i);
        CSC(cudaDeviceSynchronize());
    }
    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    float t;
    CSC(cudaEventElapsedTime(&t, start, stop));
    printf("%f\n",t);
    cudaMemcpy(mat, d_mat, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    double determinant = 1.0;
    for (int i = 0; i < n; ++i) {
        determinant *= mat[i + i * n];
    }

    if (num_swaps % 2 != 0) {
        determinant = -determinant;
    }
    std::cout.setf(std::ios::scientific);
    std::cout.precision(10);
    std::cout << determinant;

    CSC(cudaFree(d_mat));
    delete[] mat;
    return 0;
}
