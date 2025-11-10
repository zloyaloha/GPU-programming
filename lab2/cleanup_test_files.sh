#!/bin/bash

# Скрипт для очистки тестовых файлов

echo "Удаление тестовых файлов..."

# Удаляем тестовые данные
rm -f test_benchmark_results.csv
rm -f create_test_data.py

# Удаляем тестовые heatmaps
rm -rf heatmaps/

# Удаляем временные файлы бенчмарка
rm -f benchmark_results.csv
rm -f in.bin
rm -f out.bin
rm -f main

echo "Тестовые файлы удалены"
