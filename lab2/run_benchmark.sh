#!/bin/bash

# Скрипт для запуска бенчмарка CUDA и создания визуализации

echo "=== Запуск CUDA бенчмарка ==="

# Компиляция CUDA программы
echo "Компиляция CUDA программы..."
nvcc -o main main.cu

if [ $? -ne 0 ]; then
    echo "Ошибка компиляции!"
    exit 1
fi

# Создание тестового файла, если его нет
if [ ! -f "in.bin" ]; then
    echo "Создание тестового файла in.bin..."
    python3 -c "
import numpy as np
import struct

# Создаем тестовое изображение 800x600
w, h = 800, 600
data = np.random.randint(0, 255, (h, w, 4), dtype=np.uint8)

with open('in.bin', 'wb') as f:
    f.write(struct.pack('i', w))
    f.write(struct.pack('i', h))
    f.write(data.tobytes())
print(f'Создан файл in.bin размером {w}x{h}')
"
fi

# Запуск бенчмарка с перенаправлением вывода
echo "Запуск бенчмарка..."
./main > benchmark_results.csv 2>&1

if [ $? -ne 0 ]; then
    echo "Ошибка выполнения CUDA программы!"
    exit 1
fi

echo "Бенчмарк завершен. Результаты сохранены в benchmark_results.csv"

# Проверка наличия Python зависимостей
echo "Проверка Python зависимостей..."
python3 -c "import pandas, matplotlib, seaborn, numpy" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "Установка Python зависимостей..."
    pip3 install -r requirements.txt
fi

# Создание визуализации
echo "Создание визуализации..."
python3 visualize_performance.py benchmark_results.csv

echo "=== Бенчмарк завершен ==="
echo "Результаты сохранены в директории heatmaps/"
echo "Проверьте созданные PNG файлы для анализа производительности"
