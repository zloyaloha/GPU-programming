#!/usr/bin/env python3
"""
Скрипт для построения тепловой карты производительности CUDA kernel
Анализирует зависимость времени выполнения от конфигурации блоков и потоков
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

def load_and_clean_data(csv_file):
    """Загружает и очищает данные из CSV файла"""
    try:
        # Читаем CSV, пропуская последнюю строку с минимальным временем
        df = pd.read_csv(csv_file)
        
        # Удаляем строки с NaN или некорректными данными
        df = df.dropna()
        
        # Убеждаемся, что числовые колонки имеют правильный тип
        numeric_columns = ['blocks', 'threads', 'total_threads', 'avg_time_ms', 'min_time_ms', 'max_time_ms']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Удаляем строки с некорректными значениями
        df = df.dropna()
        
        print(f"Загружено {len(df)} записей")
        print(f"Диапазон блоков: {df['blocks'].min()} - {df['blocks'].max()}")
        print(f"Диапазон потоков: {df['threads'].min()} - {df['threads'].max()}")
        print(f"Диапазон времени: {df['avg_time_ms'].min():.6f} - {df['avg_time_ms'].max():.6f} мс")
        
        return df
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def create_heatmap(df, save_plots=True):
    """Создаёт тепловую карту производительности"""
    
    # Создаём pivot table для heatmap
    pivot_table = df.pivot(index='blocks', columns='threads', values='avg_time_ms')
    
    # Создаём фигуру с двумя подграфиками
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Анализ производительности CUDA Kernel', fontsize=16, fontweight='bold')
    
    # 1. Основная тепловая карта
    ax1 = axes[0]
    sns.heatmap(pivot_table, 
                annot=False,  # Не показываем числа для читаемости
                cmap='viridis_r',  # Обратный viridis (тёмный = быстрее)
                cbar_kws={'label': 'Время выполнения (мс)'},
                ax=ax1)
    ax1.set_title('Тепловая карта: Блоки vs Потоки')
    ax1.set_xlabel('Потоки в блоке')
    ax1.set_ylabel('Количество блоков')
    
    # 2. Логарифмическая шкала для лучшей видимости
    ax2 = axes[1]
    log_pivot = np.log10(pivot_table + 1e-6)  # Добавляем маленькое значение для log(0)
    sns.heatmap(log_pivot,
                annot=False,
                cmap='plasma_r',
                cbar_kws={'label': 'log10(Время выполнения) (мс)'},
                ax=ax2)
    ax2.set_title('Логарифмическая шкала')
    ax2.set_xlabel('Потоки в блоке')
    ax2.set_ylabel('Количество блоков')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('/home/zloyaloha/programming/pgrp/cuda_heatmap_analysis.png', 
                   dpi=300, bbox_inches='tight')
        print("График сохранён как: cuda_heatmap_analysis.png")
    
    plt.show()
    
    return pivot_table

def analyze_optimal_configurations(df):
    """Анализирует оптимальные конфигурации"""
    print("\n" + "="*60)
    print("АНАЛИЗ ОПТИМАЛЬНЫХ КОНФИГУРАЦИЙ")
    print("="*60)
    
    # Топ-10 конфигураций
    top_10 = df.nsmallest(10, 'avg_time_ms')
    print("\nТоп-10 конфигураций:")
    print("-" * 80)
    print(f"{'Ранг':<4} {'Блоки':<6} {'Потоки':<8} {'Всего потоков':<12} {'Время (мс)':<12} {'Разброс (мс)':<12}")
    print("-" * 80)
    
    for i, (idx, row) in enumerate(top_10.iterrows(), 1):
        spread = row['max_time_ms'] - row['min_time_ms']
        print(f"{i:<4} {int(row['blocks']):<6} {int(row['threads']):<8} "
              f"{int(row['total_threads']):<12} {row['avg_time_ms']:<12.6f} {spread:<12.6f}")
    
    # Статистика по блокам
    print(f"\nСтатистика по количеству блоков:")
    print("-" * 40)
    block_stats = df.groupby('blocks')['avg_time_ms'].agg(['min', 'mean', 'std']).round(6)
    print(block_stats)
    
    # Статистика по потокам
    print(f"\nСтатистика по потокам в блоке:")
    print("-" * 40)
    thread_stats = df.groupby('threads')['avg_time_ms'].agg(['min', 'mean', 'std']).round(6)
    print(thread_stats)

def create_detailed_heatmap(df, save_plots=True):
    """Создаёт детальную тепловую карту с аннотациями"""
    pivot_table = df.pivot(index='blocks', columns='threads', values='avg_time_ms')
    
    plt.figure(figsize=(16, 12))
    
    # Создаём маску для скрытия NaN значений
    mask = pivot_table.isna()
    
    # Тепловая карта с аннотациями для лучших значений
    sns.heatmap(pivot_table, 
                mask=mask,
                annot=True,  # Показываем значения
                fmt='.3f',   # Формат чисел
                cmap='RdYlBu_r',  # Цветовая схема (красный = медленно, синий = быстро)
                cbar_kws={'label': 'Время выполнения (мс)'},
                linewidths=0.5,
                linecolor='white')
    
    plt.title('Детальная тепловая карта производительности CUDA Kernel\n(Чем темнее цвет, тем быстрее выполнение)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Потоки в блоке', fontsize=12)
    plt.ylabel('Количество блоков', fontsize=12)
    
    # Выделяем лучшие конфигурации
    top_5 = df.nsmallest(5, 'avg_time_ms')
    for _, row in top_5.iterrows():
        plt.scatter(row['threads'] - 0.5, row['blocks'] - 0.5, 
                   s=100, color='red', marker='*', edgecolor='white', linewidth=2)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('/home/zloyaloha/programming/pgrp/cuda_detailed_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        print("Детальная тепловая карта сохранена как: cuda_detailed_heatmap.png")
    
    plt.show()

def main():
    """Основная функция"""
    csv_file = '/home/zloyaloha/programming/pgrp/table.csv'
    
    if not os.path.exists(csv_file):
        print(f"Файл {csv_file} не найден!")
        return
    
    print("Загружаем данные...")
    df = load_and_clean_data(csv_file)
    
    if df is None or df.empty:
        print("Не удалось загрузить данные!")
        return
    
    print("\nСоздаём тепловые карты...")
    
    # Основной анализ
    pivot_table = create_heatmap(df, save_plots=True)
    
    # Детальная тепловая карта
    create_detailed_heatmap(df, save_plots=True)
    
    # Анализ оптимальных конфигураций
    analyze_optimal_configurations(df)
    
    print("\nАнализ завершён! Проверьте сохранённые PNG файлы.")

if __name__ == "__main__":
    main()