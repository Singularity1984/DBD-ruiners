import numpy as np
import random
import pygame
from config import *

# Попытка импорта библиотеки noise
try:
    from noise import pnoise2
    HAS_NOISE = True
except ImportError:
    HAS_NOISE = False
    # Простая реализация шума Перлина для fallback
    def simple_noise(x, y, seed=0):
        """Простая реализация шума для fallback"""
        n = int(x * 100) + int(y * 100) * 57 + seed
        n = (n << 13) ^ n
        return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)


class StructureTemplate:
    """Шаблон конструкции (буквы Г, Т, П и т.д.)"""
    
    def __init__(self, pattern, width, height, generator_positions=None):
        """
        pattern: 2D массив, где 1 = стена, 0 = пусто
        generator_positions: список позиций генераторов относительно шаблона [(x, y), ...]
        """
        self.pattern = np.array(pattern)
        self.width = width
        self.height = height
        self.generator_positions = generator_positions or []
    
    def rotate(self, angle):
        """Поворот шаблона на угол (0, 90, 180, 270)"""
        if angle == 0:
            return self
        elif angle == 90:
            rotated = np.rot90(self.pattern, k=1)
            new_gen_pos = [(self.height - 1 - y, x) for x, y in self.generator_positions]
            return StructureTemplate(rotated, self.height, self.width, new_gen_pos)
        elif angle == 180:
            rotated = np.rot90(self.pattern, k=2)
            new_gen_pos = [(self.width - 1 - x, self.height - 1 - y) for x, y in self.generator_positions]
            return StructureTemplate(rotated, self.width, self.height, new_gen_pos)
        elif angle == 270:
            rotated = np.rot90(self.pattern, k=3)
            new_gen_pos = [(y, self.width - 1 - x) for x, y in self.generator_positions]
            return StructureTemplate(rotated, self.height, self.width, new_gen_pos)
        return self


class PerlinMapGenerator:
    """Генератор карты через шум Перлина"""
    
    def __init__(self):
        self.templates = self._create_templates()
        self.wall_size = 30  # Размер одного блока стены
    
    def _create_templates(self):
        """Создание шаблонов конструкций"""
        templates = []
        
        # Буква Г (L-образная)
        l_pattern = [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ]
        templates.append(StructureTemplate(l_pattern, 4, 4, [(2, 1)]))
        
        # Буква Т
        t_pattern = [
            [1, 1, 1, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0]
        ]
        templates.append(StructureTemplate(t_pattern, 4, 4, [(2, 2)]))
        
        # Буква П (U-образная)
        u_pattern = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        templates.append(StructureTemplate(u_pattern, 4, 4, [(2, 2)]))
        
        # Буква I (прямая)
        i_pattern = [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ]
        templates.append(StructureTemplate(i_pattern, 4, 4, [(2, 1)]))
        
        # Буква Z
        z_pattern = [
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0]
        ]
        templates.append(StructureTemplate(z_pattern, 4, 4, [(1, 2)]))
        
        # Буква H
        h_pattern = [
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 0, 1]
        ]
        templates.append(StructureTemplate(h_pattern, 4, 4, [(2, 1)]))
        
        return templates
    
    def generate_map(self, width, height, num_generators=5):
        """Генерация карты с помощью шума Перлина"""
        from environment import Wall, Generator
        
        walls = []
        generators = []
        generator_id = 0
        
        # Параметры шума Перлина
        scale = 0.05  # Масштаб шума
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        seed = random.randint(0, 1000)
        
        # Создаем сетку для размещения конструкций
        # Используем фиксированный размер ячейки сетки в пикселях
        cell_size = 120  # Размер одной ячейки сетки в пикселях
        grid_width = max(10, width // cell_size)
        grid_height = max(10, height // cell_size)
        
        # Генерируем значения шума для каждой ячейки сетки
        noise_map = np.zeros((grid_height, grid_width))
        for y in range(grid_height):
            for x in range(grid_width):
                if HAS_NOISE:
                    noise_map[y][x] = pnoise2(
                        x * scale, y * scale,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=grid_width,
                        repeaty=grid_height,
                        base=seed
                    )
                else:
                    # Fallback на простой шум
                    noise_map[y][x] = simple_noise(x * scale, y * scale, seed)
        
        # Нормализуем значения от 0 до 1
        if noise_map.max() > noise_map.min():
            noise_map = (noise_map - noise_map.min()) / (noise_map.max() - noise_map.min())
        else:
            noise_map = np.ones_like(noise_map) * 0.5
        
        # Порог для размещения конструкций (топ 50% значений) - снижен для большего количества конструкций
        threshold = 0.5
        
        placed_structures = []
        generator_positions = []
        
        # Размещаем конструкции
        for y in range(grid_height - 3):  # -3 чтобы не выходить за границы
            for x in range(grid_width - 3):
                if noise_map[y][x] > threshold:
                    # Выбираем случайный шаблон и поворот
                    template = random.choice(self.templates)
                    rotation = random.choice([0, 90, 180, 270])
                    rotated_template = template.rotate(rotation)
                    
                    # Позиция на карте
                    cell_size = 120
                    map_x = x * cell_size + 50  # Отступ от края
                    map_y = y * cell_size + 50
                    
                    # Проверяем, не пересекается ли с другими конструкциями
                    overlap = False
                    for placed in placed_structures:
                        px, py, pw, ph = placed
                        if not (map_x + rotated_template.width * self.wall_size < px or 
                               map_x > px + pw or
                               map_y + rotated_template.height * self.wall_size < py or
                               map_y > py + ph):
                            overlap = True
                            break
                    
                    if not overlap:
                        # Создаем стены из шаблона
                        for ty in range(rotated_template.height):
                            for tx in range(rotated_template.width):
                                if rotated_template.pattern[ty][tx] == 1:
                                    wall_x = map_x + tx * self.wall_size
                                    wall_y = map_y + ty * self.wall_size
                                    
                                    # Проверяем границы
                                    if (wall_x >= 50 and wall_x < width - 50 and
                                        wall_y >= 50 and wall_y < height - 50):
                                        walls.append(Wall(wall_x, wall_y, self.wall_size, self.wall_size))
                        
                        # Добавляем генераторы из шаблона
                        for gen_tx, gen_ty in rotated_template.generator_positions:
                            gen_x = map_x + gen_tx * self.wall_size + self.wall_size // 2
                            gen_y = map_y + gen_ty * self.wall_size + self.wall_size // 2
                            
                            if (gen_x >= 100 and gen_x < width - 100 and
                                gen_y >= 100 and gen_y < height - 100):
                                generator_positions.append((gen_x, gen_y, generator_id))
                                generator_id += 1
                        
                        placed_structures.append((
                            map_x, map_y,
                            rotated_template.width * self.wall_size,
                            rotated_template.height * self.wall_size
                        ))
        
        # Создаем генераторы
        # Если не хватило генераторов из шаблонов, добавляем случайные
        while len(generator_positions) < num_generators:
            attempts = 0
            while attempts < 100:
                x = random.randint(150, width - 150)
                y = random.randint(150, height - 150)
                
                # Проверяем, не в стене ли
                in_wall = False
                for wall in walls:
                    if wall.rect.collidepoint(x, y):
                        in_wall = True
                        break
                
                # Проверяем расстояние до других генераторов
                too_close = False
                for gx, gy, _ in generator_positions:
                    if (x - gx) ** 2 + (y - gy) ** 2 < 10000:  # 100^2
                        too_close = True
                        break
                
                if not in_wall and not too_close:
                    generator_positions.append((x, y, generator_id))
                    generator_id += 1
                    break
                attempts += 1
        
        # Создаем объекты генераторов
        for x, y, gid in generator_positions[:num_generators]:
            generators.append(Generator(x, y, gid))
        
        return walls, generators

