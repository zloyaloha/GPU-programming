#!/usr/bin/env python3
"""
Генератор тестовых данных для программы поиска максимума между двумя массивами.
Создаёт различные типы тестов: случайные, граничные случаи, специальные паттерны.
"""

import random
import sys
import os

def generate_random_test(n, min_val=-1000.0, max_val=1000.0, seed=None):
    """Генерирует случайные тесты"""
    if seed is not None:
        random.seed(seed)
    
    a = [random.uniform(min_val, max_val) for _ in range(n)]
    b = [random.uniform(min_val, max_val) for _ in range(n)]
    return a, b

def generate_edge_cases(n):
    """Генерирует граничные случаи"""
    test_cases = []
    
    # Случай 1: все элементы a больше b
    a1 = [i + 100.0 for i in range(n)]
    b1 = [i for i in range(n)]
    test_cases.append(("all_a_greater", a1, b1))
    
    # Случай 2: все элементы b больше a
    a2 = [i for i in range(n)]
    b2 = [i + 100.0 for i in range(n)]
    test_cases.append(("all_b_greater", a2, b2))
    
    # Случай 3: равные элементы
    a3 = [i for i in range(n)]
    b3 = [i for i in range(n)]
    test_cases.append(("equal_arrays", a3, b3))
    
    # Случай 4: чередующиеся максимумы
    a4 = [i if i % 2 == 0 else i - 50 for i in range(n)]
    b4 = [i - 50 if i % 2 == 0 else i for i in range(n)]
    test_cases.append(("alternating", a4, b4))
    
    # Случай 5: отрицательные числа
    a5 = [-i for i in range(n)]
    b5 = [-i - 10 for i in range(n)]
    test_cases.append(("negative", a5, b5))
    
    # Случай 6: очень маленькие числа
    a6 = [i * 1e-10 for i in range(n)]
    b6 = [(i + 1) * 1e-10 for i in range(n)]
    test_cases.append(("small_numbers", a6, b6))
    
    # Случай 7: очень большие числа
    a7 = [i * 1e10 for i in range(n)]
    b7 = [(i + 1) * 1e10 for i in range(n)]
    test_cases.append(("large_numbers", a7, b7))
    
    return test_cases

def generate_special_patterns(n):
    """Генерирует специальные паттерны"""
    test_cases = []
    
    # Случай 1: один массив возрастающий, другой убывающий
    a1 = [i for i in range(n)]
    b1 = [n - i for i in range(n)]
    test_cases.append(("increasing_vs_decreasing", a1, b1))
    
    # Случай 2: синусоидальные волны
    import math
    a2 = [math.sin(i * math.pi / 10) for i in range(n)]
    b2 = [math.cos(i * math.pi / 10) for i in range(n)]
    test_cases.append(("sine_vs_cosine", a2, b2))
    
    # Случай 3: ступенчатая функция
    step_size = max(1, n // 5)
    a3 = [i // step_size * 10 for i in range(n)]
    b3 = [(i // step_size + 1) * 10 for i in range(n)]
    test_cases.append(("step_function", a3, b3))
    
    return test_cases

def write_test_to_file(filename, a, b):
    """Записывает тест в файл"""
    with open(filename, 'w') as f:
        f.write(f"{len(a)}\n")
        for val in a:
            f.write(f"{val:.10e} ")
        f.write("\n")
        for val in b:
            f.write(f"{val:.10e} ")
        f.write("\n")

def generate_expected_output(a, b):
    """Генерирует ожидаемый результат"""
    return [max(a[i], b[i]) for i in range(len(a))]

def main():
    if len(sys.argv) < 2:
        print("Использование: python generate_tests.py <размер_массива> [количество_случайных_тестов]")
        print("Пример: python generate_tests.py 1000 5")
        sys.exit(1)
    
    n = int(sys.argv[1])
    num_random_tests = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    
    # Создаём директорию для тестов
    test_dir = "tests"
    os.makedirs(test_dir, exist_ok=True)
    
    test_count = 0
    
    # Генерируем случайные тесты
    print(f"Генерирую {num_random_tests} случайных тестов...")
    for i in range(num_random_tests):
        a, b = generate_random_test(n, seed=i)
        filename = f"{test_dir}/test_random_{i+1}.txt"
        write_test_to_file(filename, a, b)
        
        # Генерируем ожидаемый результат
        expected = generate_expected_output(a, b)
        expected_filename = f"{test_dir}/expected_random_{i+1}.txt"
        with open(expected_filename, 'w') as f:
            for val in expected:
                f.write(f"{val:.10e} ")
            f.write("\n")
        
        test_count += 1
        print(f"  Создан тест: {filename}")
    
    # Генерируем граничные случаи
    print("Генерирую граничные случаи...")
    edge_cases = generate_edge_cases(n)
    for name, a, b in edge_cases:
        filename = f"{test_dir}/test_edge_{name}.txt"
        write_test_to_file(filename, a, b)
        
        expected = generate_expected_output(a, b)
        expected_filename = f"{test_dir}/expected_edge_{name}.txt"
        with open(expected_filename, 'w') as f:
            for val in expected:
                f.write(f"{val:.10e} ")
            f.write("\n")
        
        test_count += 1
        print(f"  Создан тест: {filename}")
    
    # Генерируем специальные паттерны
    print("Генерирую специальные паттерны...")
    special_cases = generate_special_patterns(n)
    for name, a, b in special_cases:
        filename = f"{test_dir}/test_special_{name}.txt"
        write_test_to_file(filename, a, b)
        
        expected = generate_expected_output(a, b)
        expected_filename = f"{test_dir}/expected_special_{name}.txt"
        with open(expected_filename, 'w') as f:
            for val in expected:
                f.write(f"{val:.10e} ")
            f.write("\n")
        
        test_count += 1
        print(f"  Создан тест: {filename}")
    
    # Создаём скрипт для запуска тестов
    run_script = f"""#!/bin/bash
# Скрипт для запуска всех тестов

echo "Компиляция программы..."
g++ -O2 -std=c++17 -o main main.cpp

echo "Запуск тестов..."
for test_file in tests/test_*.txt; do
    if [[ -f "$test_file" ]]; then
        echo "Тест: $test_file"
        ./main < "$test_file"
        echo ""
    fi
done
"""
    
    with open("run_tests.sh", 'w') as f:
        f.write(run_script)
    os.chmod("run_tests.sh", 0o755)
    
    print(f"\nСоздано {test_count} тестов в директории '{test_dir}'")
    print("Для запуска всех тестов выполните: ./run_tests.sh")
    print("Для запуска конкретного теста: ./main < tests/test_random_1.txt")

if __name__ == "__main__":
    main()


