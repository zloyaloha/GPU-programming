#!/usr/bin/env python3
"""
Скрипт для визуализации результатов тестирования производительности CUDA kernel
Создает тепловые карты (heatmaps) для различных метрик производительности
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import argparse
import sys
import os

def parse_cuda_output(file_path):
    """
    Парсит CSV вывод CUDA программы
    Формат: threads,blocks,total_threads,avg_time_ms,min_time_ms,max_time_ms
    """
    try:
        # Читаем файл построчно, чтобы отделить CSV данные от строки с минимальным временем
        csv_lines = []
        min_line = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if 'MINIMAL avg_time' in line:
                    min_line = line
                elif line and not line.startswith('threads,blocks') and not line.startswith('ERROR'):  # Пропускаем заголовок и ошибки
                    # Проверяем, что строка содержит корректные CSV данные
                    # Убираем возможные ошибки CUDA в конце строки
                    clean_line = line.split('ERROR')[0].strip()
                    parts = clean_line.split(',')
                    if len(parts) >= 6:
                        try:
                            # Проверяем, что все части можно преобразовать в числа
                            int(parts[0])  # threads
                            int(parts[1])  # blocks
                            int(parts[2])  # total_threads
                            float(parts[3])  # avg_time_ms
                            float(parts[4])  # min_time_ms
                            float(parts[5])  # max_time_ms
                            csv_lines.append(clean_line)
                        except ValueError:
                            # Пропускаем строки, которые не являются корректными CSV данными
                            continue
        
        # Создаем DataFrame из CSV строк
        if not csv_lines:
            print("Не найдено корректных CSV данных в файле")
            return None, None
            
        df = pd.DataFrame([line.split(',') for line in csv_lines], columns=[
            'threads', 'blocks', 'total_threads', 'avg_time_ms', 
            'min_time_ms', 'max_time_ms'
        ])
        
        # Преобразуем типы данных
        df['threads'] = df['threads'].astype(int)
        df['blocks'] = df['blocks'].astype(int)
        df['total_threads'] = df['total_threads'].astype(int)
        df['avg_time_ms'] = df['avg_time_ms'].astype(float)
        df['min_time_ms'] = df['min_time_ms'].astype(float)
        df['max_time_ms'] = df['max_time_ms'].astype(float)
        
        return df, min_line
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return None, None

def create_heatmap_data(df):
    """
    Создает матрицы данных для тепловых карт
    """
    # Создаем pivot таблицы для различных метрик
    metrics = {
        'avg_time': df.pivot(index='threads', columns='blocks', values='avg_time_ms'),
        'min_time': df.pivot(index='threads', columns='blocks', values='min_time_ms'),
        'max_time': df.pivot(index='threads', columns='blocks', values='max_time_ms'),
        'total_threads': df.pivot(index='threads', columns='blocks', values='total_threads')
    }
    
    return metrics

def create_heatmaps(metrics, output_dir='heatmaps'):
    """
    Создает различные тепловые карты
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Настройка стиля
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # 1. Тепловая карта среднего времени выполнения
    plt.figure(figsize=(12, 10))
    sns.heatmap(metrics['avg_time'], 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Время выполнения (мс)'})
    plt.title('Среднее время выполнения CUDA kernel (мс)', fontsize=14, fontweight='bold')
    plt.xlabel('Количество блоков', fontsize=12)
    plt.ylabel('Количество потоков на блок', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/avg_time_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Тепловая карта минимального времени выполнения
    plt.figure(figsize=(12, 10))
    sns.heatmap(metrics['min_time'], 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Минимальное время (мс)'})
    plt.title('Минимальное время выполнения CUDA kernel (мс)', fontsize=14, fontweight='bold')
    plt.xlabel('Количество блоков', fontsize=12)
    plt.ylabel('Количество потоков на блок', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/min_time_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Тепловая карта максимального времени выполнения
    plt.figure(figsize=(12, 10))
    sns.heatmap(metrics['max_time'], 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Максимальное время (мс)'})
    plt.title('Максимальное время выполнения CUDA kernel (мс)', fontsize=14, fontweight='bold')
    plt.xlabel('Количество блоков', fontsize=12)
    plt.ylabel('Количество потоков на блок', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/max_time_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Тепловая карта общего количества потоков
    plt.figure(figsize=(12, 10))
    sns.heatmap(metrics['total_threads'], 
                annot=True, 
                fmt='.0f', 
                cmap='viridis',
                cbar_kws={'label': 'Общее количество потоков'})
    plt.title('Общее количество потоков (блоки × потоки)', fontsize=14, fontweight='bold')
    plt.xlabel('Количество блоков', fontsize=12)
    plt.ylabel('Количество потоков на блок', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/total_threads_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Комбинированная тепловая карта с логарифмической шкалой
    plt.figure(figsize=(12, 10))
    sns.heatmap(metrics['avg_time'], 
                annot=True, 
                fmt='.3f', 
                cmap='plasma',
                norm=LogNorm(),
                cbar_kws={'label': 'Время выполнения (мс, лог. шкала)'})
    plt.title('Среднее время выполнения (логарифмическая шкала)', fontsize=14, fontweight='bold')
    plt.xlabel('Количество блоков', fontsize=12)
    plt.ylabel('Количество потоков на блок', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/avg_time_log_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_analysis(df, min_info, output_dir='heatmaps'):
    """
    Создает дополнительный анализ производительности
    """
    # Находим лучшие конфигурации
    best_avg = df.loc[df['avg_time_ms'].idxmin()]
    best_min = df.loc[df['min_time_ms'].idxmin()]
    
    # Создаем график производительности
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # График 1: Время выполнения vs общее количество потоков
    ax1.scatter(df['total_threads'], df['avg_time_ms'], alpha=0.6, c=df['avg_time_ms'], cmap='viridis')
    ax1.set_xlabel('Общее количество потоков')
    ax1.set_ylabel('Среднее время выполнения (мс)')
    ax1.set_title('Время выполнения vs Количество потоков')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Распределение времени выполнения
    ax2.hist(df['avg_time_ms'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(best_avg['avg_time_ms'], color='red', linestyle='--', linewidth=2, label=f'Лучший: {best_avg["avg_time_ms"]:.3f} мс')
    ax2.set_xlabel('Время выполнения (мс)')
    ax2.set_ylabel('Частота')
    ax2.set_title('Распределение времени выполнения')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График 3: Соотношение блоков и потоков для лучших результатов
    top_10 = df.nsmallest(10, 'avg_time_ms')
    scatter = ax3.scatter(top_10['blocks'], top_10['threads'], 
                         c=top_10['avg_time_ms'], s=100, cmap='RdYlBu_r')
    ax3.set_xlabel('Количество блоков')
    ax3.set_ylabel('Количество потоков на блок')
    ax3.set_title('Топ-10 конфигураций (размер = время выполнения)')
    plt.colorbar(scatter, ax=ax3, label='Время выполнения (мс)')
    ax3.grid(True, alpha=0.3)
    
    # График 4: Стабильность (разброс между min и max временем)
    df['stability'] = df['max_time_ms'] - df['min_time_ms']
    ax4.scatter(df['avg_time_ms'], df['stability'], alpha=0.6, c=df['stability'], cmap='plasma')
    ax4.set_xlabel('Среднее время выполнения (мс)')
    ax4.set_ylabel('Разброс времени (мс)')
    ax4.set_title('Стабильность vs Производительность')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем отчет
    report_path = f'{output_dir}/performance_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== ОТЧЕТ О ПРОИЗВОДИТЕЛЬНОСТИ CUDA KERNEL ===\n\n")
        
        if min_info:
            f.write(f"Информация из CUDA программы:\n{min_info}\n\n")
        
        f.write("ЛУЧШИЕ КОНФИГУРАЦИИ:\n")
        f.write(f"Лучшая по среднему времени: {best_avg['blocks']} блоков, {best_avg['threads']} потоков, время: {best_avg['avg_time_ms']:.3f} мс\n")
        f.write(f"Лучшая по минимальному времени: {best_min['blocks']} блоков, {best_min['threads']} потоков, время: {best_min['min_time_ms']:.3f} мс\n\n")
        
        f.write("СТАТИСТИКА:\n")
        f.write(f"Общее количество тестированных конфигураций: {len(df)}\n")
        f.write(f"Среднее время выполнения: {df['avg_time_ms'].mean():.3f} мс\n")
        f.write(f"Стандартное отклонение: {df['avg_time_ms'].std():.3f} мс\n")
        f.write(f"Минимальное время: {df['avg_time_ms'].min():.3f} мс\n")
        f.write(f"Максимальное время: {df['avg_time_ms'].max():.3f} мс\n")
        f.write(f"Медиана: {df['avg_time_ms'].median():.3f} мс\n\n")
        
        f.write("ТОП-5 КОНФИГУРАЦИЙ:\n")
        top_5 = df.nsmallest(5, 'avg_time_ms')
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            f.write(f"{i}. {row['blocks']} блоков × {row['threads']} потоков = {row['total_threads']} потоков, время: {row['avg_time_ms']:.3f} мс\n")

def main():
    parser = argparse.ArgumentParser(description='Визуализация результатов тестирования CUDA kernel')
    parser.add_argument('input_file', help='Путь к CSV файлу с результатами тестирования')
    parser.add_argument('-o', '--output', default='heatmaps', help='Директория для сохранения графиков (по умолчанию: heatmaps)')
    parser.add_argument('--no-analysis', action='store_true', help='Не создавать дополнительный анализ')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Ошибка: Файл {args.input_file} не найден")
        sys.exit(1)
    
    print("Парсинг данных...")
    df, min_info = parse_cuda_output(args.input_file)
    
    if df is None:
        print("Ошибка при парсинге данных")
        sys.exit(1)
    
    print(f"Загружено {len(df)} записей")
    print("Создание тепловых карт...")
    
    # Создаем метрики
    metrics = create_heatmap_data(df)
    
    # Создаем тепловые карты
    create_heatmaps(metrics, args.output)
    
    if not args.no_analysis:
        print("Создание дополнительного анализа...")
        create_performance_analysis(df, min_info, args.output)
    
    print(f"Графики сохранены в директории: {args.output}")
    print("Созданные файлы:")
    for file in os.listdir(args.output):
        print(f"  - {file}")

if __name__ == "__main__":
    main()
