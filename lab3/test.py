import struct
import numpy as np

def generate_test_file(filename, width=256, height=256, num_classes=4):
    """
    Генерирует бинарный файл в формате:
    [int width][int height][uchar4 * (width*height)]
    где uchar4 = (R, G, B, A)
    """

    # Создаём пустое изображение RGBA
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Зададим области с разными цветами (по классам)
    colors = [
        (255, 0, 0, 255),     # красный
        (0, 255, 0, 255),     # зелёный
        (0, 0, 255, 255),     # синий
        (255, 255, 0, 255),   # жёлтый
    ]

    block_w = width // num_classes
    for i in range(num_classes):
        x_start = i * block_w
        x_end = (i + 1) * block_w if i < num_classes - 1 else width
        img[:, x_start:x_end, :] = colors[i % len(colors)]

    # Сохраняем в бинарный файл
    with open(filename, "wb") as f:
        # Записываем ширину и высоту (int32)
        f.write(struct.pack("ii", width, height))
        # Пиксели по порядку (в row-major)
        f.write(img.tobytes())

    print(f"✅ Файл '{filename}' успешно создан ({width}x{height})")


if __name__ == "__main__":
    generate_test_file("in.data", width=9000, height=9000, num_classes=4)
