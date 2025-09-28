#include <iostream>
#include <chrono>
#include <vector>

std::vector<double> find_max_between_two_arrays(const std::vector<double>& a, const std::vector<double>& b)
{
    std::vector<double> res(a.size());
    for (int i = 0; i < a.size(); ++i) {
        res[i] = std::max(a[i], b[i]);
    }
    return res;
}

int main() {
    int n;
    std::cin >> n;
    std::vector<double> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        std::cin >> a[i];
    }
    for (int i = 0; i < n; ++i) {
        std::cin >> b[i];
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> res = find_max_between_two_arrays(a, b);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Время выполнения: " << diff.count() * 1000 << " миллисекунд" << std::endl;
}