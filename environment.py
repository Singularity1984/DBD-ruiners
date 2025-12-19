import pygame
import numpy as np
import random
import math
from config import *


class Wall:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, WALL_COLOR, self.rect)
        pygame.draw.rect(surface, (60, 60, 80), self.rect, 2)


class Generator:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.progress = 0
        self.fixed = False
        self.radius = 25
        self.repair_radius = 35

    def draw(self, surface):
        color = (0, 200, 0) if self.fixed else GENERATOR_COLOR
        pygame.draw.circle(surface, color, (self.x, self.y), self.radius)
        if not self.fixed:
            pygame.draw.circle(surface, (100, 100, 100), (self.x, self.y), self.radius, 2)
            progress_angle = int(360 * (self.progress / 100))
            if progress_angle > 0:
                pygame.draw.arc(surface, (0, 255, 0),
                                (self.x - self.radius, self.y - self.radius,
                                 self.radius * 2, self.radius * 2),
                                0, np.radians(progress_angle), 4)


class PerlinNoise:
    """Простая реализация Perlin noise для генерации карт"""
    
    seed_offset = 0  # Смещение seed для разнообразия карт
    
    @staticmethod
    def set_seed(seed):
        """Установить seed для генерации"""
        PerlinNoise.seed_offset = seed % 1000000 if seed is not None else 0
    
    @staticmethod
    def noise(x, y):
        """Простой 2D шум (упрощенная версия Perlin noise)"""
        # Добавляем seed_offset для разнообразия
        n = int(x) + int(y) * 57 + PerlinNoise.seed_offset
        n = (n << 13) ^ n
        return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0)
    
    @staticmethod
    def smooth_noise(x, y):
        """Сглаженный шум"""
        corners = (PerlinNoise.noise(x-1, y-1) + PerlinNoise.noise(x+1, y-1) + 
                  PerlinNoise.noise(x-1, y+1) + PerlinNoise.noise(x+1, y+1)) / 16.0
        sides = (PerlinNoise.noise(x-1, y) + PerlinNoise.noise(x+1, y) + 
                PerlinNoise.noise(x, y-1) + PerlinNoise.noise(x, y+1)) / 8.0
        center = PerlinNoise.noise(x, y) / 4.0
        return corners + sides + center
    
    @staticmethod
    def interpolated_noise(x, y):
        """Интерполированный шум"""
        integer_X = int(x)
        fractional_X = x - integer_X
        
        integer_Y = int(y)
        fractional_Y = y - integer_Y
        
        v1 = PerlinNoise.smooth_noise(integer_X, integer_Y)
        v2 = PerlinNoise.smooth_noise(integer_X + 1, integer_Y)
        v3 = PerlinNoise.smooth_noise(integer_X, integer_Y + 1)
        v4 = PerlinNoise.smooth_noise(integer_X + 1, integer_Y + 1)
        
        i1 = PerlinNoise.interpolate(v1, v2, fractional_X)
        i2 = PerlinNoise.interpolate(v3, v4, fractional_X)
        
        return PerlinNoise.interpolate(i1, i2, fractional_Y)
    
    @staticmethod
    def interpolate(a, b, x):
        """Косинусная интерполяция"""
        ft = x * math.pi
        f = (1 - math.cos(ft)) * 0.5
        return a * (1 - f) + b * f
    
    @staticmethod
    def perlin_noise(x, y, octaves=4, persistence=0.5, scale=0.01):
        """Perlin noise с несколькими октавами"""
        total = 0
        frequency = scale
        amplitude = 1
        max_value = 0
        
        for i in range(octaves):
            total += PerlinNoise.interpolated_noise(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2
        
        return total / max_value


class EnvironmentGenerator:
    """Генератор игрового окружения с использованием Perlin noise"""

    @staticmethod
    def create_training_room():
        """Преднастроенная комната: два выживших, три генератора, один охотник"""
        walls = [Wall(x, y, w, h) for x, y, w, h in TRAINING_ROOM_LAYOUT["walls"]]
        
        # Создаем генераторы: сверху, в центре и снизу
        if "generators" in TRAINING_ROOM_LAYOUT:
            # Используем список генераторов из конфигурации
            generators = [Generator(pos[0], pos[1], idx) for idx, pos in enumerate(TRAINING_ROOM_LAYOUT["generators"])]
            generator_pos = TRAINING_ROOM_LAYOUT["generator"]  # Оставляем для обратной совместимости
        else:
            # Обратная совместимость со старой конфигурацией
            generator_pos = TRAINING_ROOM_LAYOUT["generator"]
            generators = [Generator(generator_pos[0], generator_pos[1], 0)]
        
        exits = TRAINING_ROOM_LAYOUT["exits"]
        survivors = TRAINING_ROOM_LAYOUT["survivors"]
        hunter = TRAINING_ROOM_LAYOUT["hunter"]

        return {
            "walls": walls,
            "generators": generators,
            "generator_pos": generator_pos,
            "exits": exits,
            "survivors": survivors,
            "hunter": hunter
        }

    @staticmethod
    def create_perlin_walls(seed=None):
        """Генерация стен на основе Perlin noise"""
        # Устанавливаем seed для Perlin noise
        PerlinNoise.set_seed(seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        walls = []
        # Размер сетки для генерации
        grid_size = 20
        cell_width = WIDTH // grid_size
        cell_height = HEIGHT // grid_size
        
        # Создаем карту шума
        noise_map = np.zeros((grid_size, grid_size))
        offset_x = random.uniform(0, 1000)
        offset_y = random.uniform(0, 1000)
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * cell_width + offset_x
                y = j * cell_height + offset_y
                noise_value = PerlinNoise.perlin_noise(x, y, octaves=4, persistence=0.5, scale=0.05)
                noise_map[i][j] = noise_value
        
        # Порог для создания стен (только высокие значения шума)
        wall_threshold = 0.3
        
        # Объединяем соседние ячейки в стены
        visited = np.zeros((grid_size, grid_size), dtype=bool)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if noise_map[i][j] > wall_threshold and not visited[i][j]:
                    # Начинаем новую стену
                    wall_cells = [(i, j)]
                    visited[i][j] = True
                    queue = [(i, j)]
                    
                    # Объединяем соседние ячейки
                    while queue:
                        ci, cj = queue.pop(0)
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if (0 <= ni < grid_size and 0 <= nj < grid_size and 
                                not visited[ni][nj] and noise_map[ni][nj] > wall_threshold):
                                visited[ni][nj] = True
                                wall_cells.append((ni, nj))
                                queue.append((ni, nj))
                    
                    # Создаем прямоугольную стену из группы ячеек
                    if len(wall_cells) > 0:
                        min_i = min(c[0] for c in wall_cells)
                        max_i = max(c[0] for c in wall_cells)
                        min_j = min(c[1] for c in wall_cells)
                        max_j = max(c[1] for c in wall_cells)
                        
                        # Размеры стены с небольшим запасом
                        wall_width = (max_i - min_i + 1) * cell_width
                        wall_height = (max_j - min_j + 1) * cell_height
                        wall_x = min_i * cell_width
                        wall_y = min_j * cell_height
                        
                        # Проверяем, что стена не блокирует центр
                        wall_rect = pygame.Rect(wall_x, wall_y, wall_width, wall_height)
                        center_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 - 150, 300, 300)
                        
                        if not wall_rect.colliderect(center_rect) and wall_width > 20 and wall_height > 20:
                            walls.append(Wall(wall_x, wall_y, wall_width, wall_height))
        
        # Если получилось мало стен, добавляем случайные
        if len(walls) < NUM_WALLS[0]:
            for _ in range(NUM_WALLS[0] - len(walls)):
                width = random.randint(30, 60)
                height = random.randint(30, 60)
                x = random.randint(100, WIDTH - width - 100)
                y = random.randint(100, HEIGHT - height - 100)
                
                wall_rect = pygame.Rect(x, y, width, height)
                center_rect = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 - 150, 300, 300)
                
                if not wall_rect.colliderect(center_rect):
                    walls.append(Wall(x, y, width, height))
        
        return walls
    
    @staticmethod
    def create_random_walls(count=None):
        """Старый метод (оставлен для совместимости)"""
        if count is None:
            count = random.randint(NUM_WALLS[0], NUM_WALLS[1])

        walls = []
        for _ in range(count):
            width = random.randint(30, 60)
            height = random.randint(30, 60)
            x = random.randint(100, WIDTH - width - 100)
            y = random.randint(100, HEIGHT - height - 100)

            wall_rect = pygame.Rect(x, y, width, height)
            center_rect = pygame.Rect(WIDTH // 2 - 100, HEIGHT // 2 - 100, 200, 200)

            if not wall_rect.colliderect(center_rect):
                walls.append(Wall(x, y, width, height))

        return walls

    @staticmethod
    def create_random_generators(count=None, min_distance=100, walls=[]):
        if count is None:
            count = random.randint(NUM_GENERATORS[0], NUM_GENERATORS[1])

        generators = []
        attempts = 0
        max_attempts = 200
        min_distance_sq = min_distance ** 2

        while len(generators) < count and attempts < max_attempts:
            attempts += 1
            x = random.randint(150, WIDTH - 150)
            y = random.randint(150, HEIGHT - 150)

            in_wall = False
            for wall in walls:
                if wall.rect.collidepoint(x, y):
                    in_wall = True
                    break

            if in_wall:
                continue

            too_close = False
            for existing_gen in generators:
                distance_sq = (existing_gen.x - x) ** 2 + (existing_gen.y - y) ** 2
                if distance_sq < min_distance_sq:
                    too_close = True
                    break

            if not too_close:
                generators.append(Generator(x, y, len(generators)))

        while len(generators) < count:
            x = random.randint(150, WIDTH - 150)
            y = random.randint(150, HEIGHT - 150)
            generators.append(Generator(x, y, len(generators)))

        return generators

    @staticmethod
    def create_random_exits(count=None, walls=[]):
        if count is None:
            count = random.randint(NUM_EXITS[0], NUM_EXITS[1])

        exits = []
        edge_positions = []

        for i in range(20):
            edge_positions.append((random.randint(100, WIDTH - 100), 50))
            edge_positions.append((random.randint(100, WIDTH - 100), HEIGHT - 50))

        for i in range(20):
            edge_positions.append((50, random.randint(100, HEIGHT - 100)))
            edge_positions.append((WIDTH - 50, random.randint(100, HEIGHT - 100)))

        random.shuffle(edge_positions)

        for pos in edge_positions:
            if len(exits) >= count:
                break

            in_wall = False
            for wall in walls:
                closest_x = max(wall.rect.left, min(pos[0], wall.rect.right))
                closest_y = max(wall.rect.top, min(pos[1], wall.rect.bottom))
                distance_sq = (pos[0] - closest_x) ** 2 + (pos[1] - closest_y) ** 2
                if distance_sq < 2500:
                    in_wall = True
                    break

            if not in_wall and pos not in exits:
                exits.append(pos)

        while len(exits) < count:
            side = random.choice(['top', 'bottom', 'left', 'right'])
            if side == 'top':
                exits.append((random.randint(100, WIDTH - 100), 50))
            elif side == 'bottom':
                exits.append((random.randint(100, WIDTH - 100), HEIGHT - 50))
            elif side == 'left':
                exits.append((50, random.randint(100, HEIGHT - 100)))
            else:
                exits.append((WIDTH - 50, random.randint(100, HEIGHT - 100)))

        return exits