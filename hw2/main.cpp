#include <iostream>
#include <algorithm>
#include <cstdlib>

void bubble_sort(float *arr, int size)
{
    for (int i = 0; i < size - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < size - 1 - i; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

int main() {
    std::cout << std::scientific;
    std::cout.precision(6);

    int size;
    std::cin >> size;

    float* arr = (float *)malloc(size * sizeof(float));
    
    for (int i = 0; i < size; ++i) {
        std::cin >> arr[i];
    }

    bubble_sort(arr, size);

    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << ' ';
    }
    std::cout << std::endl;
    
    std::free(arr);
}