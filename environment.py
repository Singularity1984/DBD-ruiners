import pygame
import numpy as np
import random
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


class EnvironmentGenerator:
    """Генератор игрового окружения"""

    @staticmethod
    def create_random_walls(count=None):
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