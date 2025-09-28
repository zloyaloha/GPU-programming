#include <iostream>
#include <cmath>
#include <limits>

static int solveQuadratic(float a, float b, float c, float &x1, float &x2) {
    const float eps = 1e-12;

    if (std::abs(a) < eps) {
        if (std::abs(b) < eps) {
            if (std::abs(c) < eps) {
                return -1;
            } else {
                return 0;
            }
        }
        x1 = x2 = -c / b;
        return 1;
    }

    const float d = b * b - 4.0 * a * c;

    if (d > eps) {
        const float sqrtD = std::sqrt(d);
        x1 = (-b + sqrtD) / (2.0 * a);
        x2 = (-b - sqrtD) / (2.0 * a);
        return 2;
    } else if (std::abs(d) <= eps) {
        x1 = x2 = -b / (2.0 * a);
        return 1;
    } else if (d < 0) {
        return -2;
    }

    return 0;
}

int main() {
    std::cout.precision(6);

    float a, b, c;
    if (!(std::cin >> a >> b >> c)) {
        std::cerr << "Ошибка ввода" << std::endl;
        return 1;
    }

    float x1 = 0, x2 = 0;
    int count = solveQuadratic(a, b, c, x1, x2);

    if (count == -2) {
        std::cout << "imaginary" << std::endl;
    } else if (count == -1) {
        std::cout << "any" << std::endl;
    } else if (count == 0) {
        std::cout << "incorrect" << std::endl;
    } else if (count == 1) {
        std::cout << std::fixed << x1 << std::endl;
    } else {
        std::cout << std::fixed << x1 << ' ' << x2 << std::endl;
    }

    return 0;
}