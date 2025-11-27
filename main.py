import pygame
import numpy as np
import random
import pickle
import os
import time
from collections import defaultdict
import logging
import weakref
from config import *  # Импортируем все настройки

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DBD_QL')

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dead by Daylight Q-Learning - OPTIMIZED")
clock = pygame.time.Clock()


class SmartQTable:
    """Оптимизированная Q-table с LRU кэшированием"""

    def __init__(self, max_size=MAX_Q_TABLE_SIZE):
        self._data = {}
        self.max_size = max_size
        self.access_order = []  # для LRU кэша

    def __getitem__(self, key):
        if key in self._data:
            # Обновляем порядок использования
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self._data[key]
        else:
            if len(self._data) >= self.max_size:
                # Удаляем наименее используемый элемент
                lru_key = self.access_order.pop(0)
                del self._data[lru_key]
            self._data[key] = [0.0] * 5  # 5 действий
            self.access_order.append(key)
            return self._data[key]

    def __setitem__(self, key, value):
        if key not in self._data and len(self._data) >= self.max_size:
            lru_key = self.access_order.pop(0)
            del self._data[lru_key]

        self._data[key] = value
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def __contains__(self, key):
        return key in self._data

    def clear(self):
        self._data.clear()
        self.access_order.clear()

    def update(self, other):
        for k, v in other.items():
            self[k] = v

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __len__(self):
        return len(self._data)

    def prune_infrequent(self, min_visits=3):
        """Удаляет редко используемые состояния"""
        if len(self._data) < self.max_size * 0.7:
            return 0

        keys_to_remove = [k for k in self.access_order
                          if self.access_order.count(k) < min_visits]

        for key in keys_to_remove:
            if key in self._data:
                del self._data[key]
            while key in self.access_order:
                self.access_order.remove(key)

        return len(keys_to_remove)


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


class Agent:
    def __init__(self, x, y, color, is_hunter=False, agent_id=0):
        self.x = x
        self.y = y
        self.color = color
        self.speed = HUNTER_SPEED if is_hunter else SURVIVOR_SPEED
        self.radius = HUNTER_RADIUS if is_hunter else SURVIVOR_RADIUS
        self.is_hunter = is_hunter
        self.agent_id = agent_id

        # Оптимизированная Q-table
        self.q_table = SmartQTable(MAX_Q_TABLE_SIZE)
        self.epsilon = 1.0
        self.caught = False
        self.escaped = False
        self.fixing_generator = False
        self.cooldown = 0
        self.last_rewards = []

        # Система зрения
        self.vision_direction = 0
        self.vision_radius = HUNTER_VISION_RADIUS if is_hunter else SURVIVOR_VISION_RADIUS
        self.vision_angle = HUNTER_VISION_ANGLE if is_hunter else SURVIVOR_VISION_ANGLE
        self.visible_generators = []
        self.visible_survivors = []
        self.visible_hunter = None
        self.visible_exits = []
        self.last_action = 3

        # Система захвата
        self.capture_target = None
        self.capture_progress = 0
        self.hold_steps = 0
        self.last_distance_to_target = float('inf')

        # Статистика для оптимизации
        self.state_visits = defaultdict(int)

        # Улучшенная система отслеживания позиций
        self.position_history = []
        self.stuck_counter = 0
        self.consecutive_wall_hits = 0
        self.escape_mode = False
        self.escape_steps = 0
        self.movement_pattern = []

    def update_vision(self, other_agents, generators, exits, walls):
        self.visible_generators = []
        self.visible_survivors = []
        self.visible_hunter = None
        self.visible_exits = []

        if self.last_action == 0:  # up
            self.vision_direction = 90
        elif self.last_action == 1:  # down
            self.vision_direction = 270
        elif self.last_action == 2:  # left
            self.vision_direction = 180
        elif self.last_action == 3:  # right
            self.vision_direction = 0

        for gen in generators:
            if self.is_object_in_vision(gen.x, gen.y, gen.radius):
                self.visible_generators.append(gen)

        for agent in other_agents:
            if agent != self:
                if self.is_object_in_vision(agent.x, agent.y, agent.radius):
                    if agent.is_hunter:
                        self.visible_hunter = agent
                    else:
                        self.visible_survivors.append(agent)

        for exit_pos in exits:
            if self.is_object_in_vision(exit_pos[0], exit_pos[1], 20):
                self.visible_exits.append(exit_pos)

    def is_object_in_vision(self, obj_x, obj_y, obj_radius=0):
        dx = obj_x - self.x
        dy = obj_y - self.y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        if distance > self.vision_radius + obj_radius:
            return False

        angle_to_obj = np.degrees(np.arctan2(dy, dx))
        angle_to_obj = (angle_to_obj + 360) % 360
        if angle_to_obj > 180:
            angle_to_obj -= 360

        angle_diff = abs(angle_to_obj - self.vision_direction)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff <= self.vision_angle / 2

    def draw_vision_cone(self, surface):
        if self.caught or self.escaped:
            return

        cone_surface = pygame.Surface((self.vision_radius * 2, self.vision_radius * 2), pygame.SRCALPHA)

        start_angle = self.vision_direction - self.vision_angle / 2
        end_angle = self.vision_direction + self.vision_angle / 2

        points = []
        num_segments = 60

        center = (self.vision_radius, self.vision_radius)

        for i in range(num_segments + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_segments
            rad = np.radians(angle)
            x = self.vision_radius + self.vision_radius * np.cos(rad)
            y = self.vision_radius + self.vision_radius * np.sin(rad)
            points.append((x, y))

        for i in range(len(points) - 1):
            segment_points = [center, points[i], points[i + 1]]

            dist_factor = min(1.0, np.sqrt(
                (points[i][0] - center[0]) ** 2 + (points[i][1] - center[1]) ** 2) / self.vision_radius)
            alpha = int(80 * (1 - dist_factor * 0.7))

            if self.is_hunter:
                color = (255, 100, 100, alpha)
            else:
                color = (100, 100, 255, alpha)

            if len(segment_points) >= 3:
                pygame.draw.polygon(cone_surface, color, segment_points)

        outline_points = points
        if len(outline_points) > 1:
            if self.is_hunter:
                outline_color = (255, 50, 50, 150)
            else:
                outline_color = (50, 50, 255, 150)
            pygame.draw.lines(cone_surface, outline_color, False, outline_points, 2)

        surface.blit(cone_surface, (self.x - self.vision_radius, self.y - self.vision_radius))

        end_x = self.x + self.vision_radius * 0.8 * np.cos(np.radians(self.vision_direction))
        end_y = self.y + self.vision_radius * 0.8 * np.sin(np.radians(self.vision_direction))

        if self.is_hunter:
            direction_color = (255, 100, 100, 200)
        else:
            direction_color = (100, 100, 255, 200)

        pygame.draw.line(surface, direction_color, (self.x, self.y), (end_x, end_y), 3)
        pygame.draw.circle(surface, direction_color, (int(end_x), int(end_y)), 4)

    def get_optimized_state(self, other_agents, generators, exits, generators_fixed):
        """Оптимизированная система состояний с бинаризованными расстояниями"""
        self.update_vision(other_agents, generators, exits, [])

        # Индикатор застревания
        stuck_indicator = 1 if self.escape_mode else 0

        if self.is_hunter:
            # Состояние для охотника
            active_survivors = [a for a in other_agents if not a.is_hunter and not a.escaped and not a.caught]

            if not active_survivors:
                return (2, 2, 0, stuck_indicator)  # Нет целей

            nearest = min(active_survivors, key=lambda a: (a.x - self.x) ** 2 + (a.y - self.y) ** 2)
            distance = np.sqrt((nearest.x - self.x) ** 2 + (nearest.y - self.y) ** 2)

            # Бинаризованное расстояние (3 уровня)
            if distance < 100:
                dist_state = 0  # Близко
            elif distance < 300:
                dist_state = 1  # Средне
            else:
                dist_state = 2  # Далеко

            # Направление к цели
            dx = 1 if nearest.x > self.x else 0
            dy = 1 if nearest.y > self.y else 0

            # Прогресс захвата
            if self.capture_target and self.capture_target in active_survivors:
                capture_dist = np.sqrt((self.capture_target.x - self.x) ** 2 + (self.capture_target.y - self.y) ** 2)
                if capture_dist < CAPTURE_RADIUS:
                    capture_progress = min(2, int(self.hold_steps / (CAPTURE_HOLD_STEPS / 3)))
                    return (dx, dy, capture_progress, stuck_indicator)

            return (dx, dy, dist_state, stuck_indicator)

        else:
            # Состояние для выжившего
            hunter_visible = 1 if self.visible_hunter else 0

            # Проверка захвата охотником
            hunter = next((a for a in other_agents if a.is_hunter), None)
            if hunter and hunter.capture_target == self:
                distance = np.sqrt((hunter.x - self.x) ** 2 + (hunter.y - self.y) ** 2)
                if distance < CAPTURE_RADIUS:
                    capture_state = min(2, int(hunter.hold_steps / (CAPTURE_HOLD_STEPS / 3)))
                    return (hunter_visible, 1, capture_state, stuck_indicator)

            # Приоритеты: генераторы -> выходы -> бегство
            if generators_fixed < len(generators) and self.visible_generators:
                nearest_gen = min(self.visible_generators, key=lambda g: (g.x - self.x) ** 2 + (g.y - self.y) ** 2)
                distance_to_gen = np.sqrt((nearest_gen.x - self.x) ** 2 + (nearest_gen.y - self.y) ** 2)

                if distance_to_gen < 100:
                    gen_state = 0  # Очень близко
                elif distance_to_gen < 250:
                    gen_state = 1  # Близко
                else:
                    gen_state = 2  # Далеко

                return (hunter_visible, gen_state, 0, stuck_indicator)

            elif self.visible_exits and generators_fixed == len(generators):
                nearest_exit = min(self.visible_exits, key=lambda e: (e[0] - self.x) ** 2 + (e[1] - self.y) ** 2)
                distance_to_exit = np.sqrt((nearest_exit[0] - self.x) ** 2 + (nearest_exit[1] - self.y) ** 2)

                if distance_to_exit < 100:
                    exit_state = 0
                elif distance_to_exit < 250:
                    exit_state = 1
                else:
                    exit_state = 2

                return (hunter_visible, exit_state, 1, stuck_indicator)

            else:
                # Режим выживания/поиска
                return (hunter_visible, 2, 2, stuck_indicator)

    def choose_action(self, state, epsilon):
        # Увеличиваем случайность при застревании
        current_epsilon = epsilon
        if self.escape_mode:
            current_epsilon = min(1.0, epsilon + ESCAPE_BOOST)

        if random.uniform(0, 1) < current_epsilon:
            action = random.randint(0, 4)
        else:
            action = np.argmax(self.q_table[state])

        self.last_action = action
        self.state_visits[state] += 1
        return action

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
        self.q_table[state][action] = new_q
        self.last_rewards.append(reward)
        if len(self.last_rewards) > 100:
            self.last_rewards.pop(0)

    def _check_wall_collision(self, x, y, walls):
        """Улучшенная проверка коллизий"""
        new_rect = pygame.Rect(x - self.radius, y - self.radius,
                               self.radius * 2, self.radius * 2)

        for wall in walls:
            if new_rect.colliderect(wall.rect):
                return True
        return False

    def _is_truly_stuck(self):
        """Более точное определение застревания"""
        if len(self.position_history) < STUCK_THRESHOLD:
            return False

        # Проверяем разнообразие позиций
        recent = self.position_history[-STUCK_THRESHOLD:]
        unique_positions = len(set((int(x / 10), int(y / 10)) for x, y in recent))

        # Если позиции почти не меняются
        if unique_positions < 5:
            return True

        # Проверяем циклическое движение
        if len(self.movement_pattern) >= 10:
            # Если повторяются одни и те же движения
            if len(set(self.movement_pattern)) < 4:
                return True

        # Проверяем общее движение
        total_movement = 0
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i - 1][0]
            dy = recent[i][1] - recent[i - 1][1]
            total_movement += np.sqrt(dx ** 2 + dy ** 2)

        avg_movement = total_movement / (len(recent) - 1)
        return avg_movement < 1.0 or self.consecutive_wall_hits > 8

    def smooth_move(self, action, agents, generators, walls):
        """Улучшенное движение с разделением осей"""
        if self.caught or self.escaped:
            return 0

        old_x, old_y = self.x, self.y
        reward = -0.05

        current_speed = self.speed
        if not self.is_hunter and self.cooldown > 0:
            current_speed *= 0.7
            self.cooldown -= 1

        # Двигаемся по осям отдельно для лучшего обхода препятствий
        new_x, new_y = self.x, self.y

        if action == 0:  # up
            new_y -= current_speed
        elif action == 1:  # down
            new_y += current_speed
        elif action == 2:  # left
            new_x -= current_speed
        elif action == 3:  # right
            new_x += current_speed

        # Проверяем границы
        new_x = max(self.radius, min(WIDTH - self.radius, new_x))
        new_y = max(self.radius, min(HEIGHT - self.radius, new_y))

        # Пробуем двигаться по X
        temp_x = new_x
        temp_y = self.y
        x_collision = self._check_wall_collision(temp_x, temp_y, walls)

        if not x_collision:
            self.x = temp_x
            self.consecutive_wall_hits = 0
        else:
            reward -= 0.3
            self.consecutive_wall_hits += 1

        # Пробуем двигаться по Y
        temp_x = self.x
        temp_y = new_y
        y_collision = self._check_wall_collision(temp_x, temp_y, walls)

        if not y_collision:
            self.y = temp_y
            self.consecutive_wall_hits = 0
        else:
            reward -= 0.3
            self.consecutive_wall_hits += 1

        # Обновляем историю позиций и паттернов движения
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > POSITION_MEMORY:
            self.position_history.pop(0)

        # Записываем паттерн движения
        if len(self.position_history) >= 2:
            dx = self.position_history[-1][0] - self.position_history[-2][0]
            dy = self.position_history[-1][1] - self.position_history[-2][1]
            movement = (np.sign(dx), np.sign(dy))
            self.movement_pattern.append(movement)
            if len(self.movement_pattern) > 20:
                self.movement_pattern.pop(0)

        # Проверяем застревание
        if self._is_truly_stuck():
            if not self.escape_mode:
                self.escape_mode = True
                self.escape_steps = 0
                reward -= 8  # Значительный штраф за застревание
                #logger.info(f"Агент {self.agent_id} застрял! Активирован режим побега")
            else:
                self.escape_steps += 1
                if self.escape_steps > 30:
                    reward -= 3  # Дополнительный штраф за долгое застревание
        else:
            if self.escape_mode:
                self.escape_mode = False
                reward += 15  # Награда за выход из ловушки
                #logger.info(f"Агент {self.agent_id} вышел из ловушки!")
            self.escape_steps = 0

        self.update_vision(agents, generators, [], walls)

        # Логика наград для охотника
        if self.is_hunter:
            active_survivors = [a for a in agents if not a.is_hunter and not a.escaped and not a.caught]

            if active_survivors:
                nearest = min(active_survivors, key=lambda a: (a.x - self.x) ** 2 + (a.y - self.y) ** 2)
                current_distance = np.sqrt((nearest.x - self.x) ** 2 + (nearest.y - self.y) ** 2)

                if hasattr(self, 'last_distance_to_target') and self.last_distance_to_target != float('inf'):
                    if current_distance < self.last_distance_to_target:
                        reward += 2
                    elif current_distance > self.last_distance_to_target + 10:
                        reward -= 2

                self.last_distance_to_target = current_distance

                if current_distance < CAPTURE_RADIUS:
                    if self.capture_target != nearest:
                        self.capture_target = nearest
                        self.hold_steps = 0
                        reward += 8

                    self.hold_steps += 1
                    reward += 3

                    if current_distance < CAPTURE_RADIUS * 0.6:
                        reward += 2

                    if self.hold_steps >= CAPTURE_HOLD_STEPS:
                        self.capture_target.caught = True
                        reward += 120
                        self.capture_target = None
                        self.hold_steps = 0
                else:
                    if self.capture_target == nearest:
                        reward -= 6
                        self.capture_target = None
                        self.hold_steps = 0

                if self.visible_survivors:
                    reward += 1
            else:
                self.capture_target = None
                self.hold_steps = 0

        # Логика наград для выжившего
        else:
            self.fixing_generator = False

            hunter = next((a for a in agents if a.is_hunter), None)
            if hunter:
                hunter_distance = np.sqrt((hunter.x - self.x) ** 2 + (hunter.y - self.y) ** 2)
                old_hunter_dist = np.sqrt((hunter.x - old_x) ** 2 + (hunter.y - old_y) ** 2)

                if hunter_distance > old_hunter_dist + 10:
                    reward += 4
                elif hunter_distance < old_hunter_dist - 5:
                    reward -= 3

                if hunter_distance < CAPTURE_RADIUS:
                    reward -= 3
                    if hunter.capture_target == self:
                        reward -= 4

                if hunter_distance > CAPTURE_RADIUS and old_hunter_dist < CAPTURE_RADIUS:
                    reward += 30

            if self.visible_generators:
                reward += 2

            for generator in self.visible_generators:
                if not generator.fixed:
                    distance_sq = (generator.x - self.x) ** 2 + (generator.y - self.y) ** 2

                    old_gen_dist_sq = (generator.x - old_x) ** 2 + (generator.y - old_y) ** 2
                    if distance_sq < old_gen_dist_sq:
                        reward += 4

                    if distance_sq < generator.repair_radius ** 2:
                        if action == 4:  # Действие починки
                            self.fixing_generator = True
                            generator.progress += 10
                            if generator.progress >= 100:
                                generator.fixed = True
                                reward += 250
                                self.cooldown = 10
                            else:
                                reward += 12
                        else:
                            reward -= 0.5
                    else:
                        if action == 4:
                            reward -= 0.5

            if self.escaped:
                reward += 150

            if self.caught:
                reward -= 80

        return reward

    def prune_q_table(self, min_visits=3):
        """Улучшенная очистка Q-table"""
        if len(self.q_table) < MAX_Q_TABLE_SIZE * 0.7:
            return 0

        # Используем встроенную очистку SmartQTable
        removed = self.q_table.prune_infrequent(min_visits)

        # Дополнительная очистка state_visits
        keys_to_remove = [state for state in self.state_visits
                          if self.state_visits[state] < min_visits]
        for state in keys_to_remove:
            del self.state_visits[state]

        logger.info(f"Агент {self.agent_id}: удалено {removed} состояний, осталось {len(self.q_table)}")
        return removed


# Вспомогательные функции

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


def create_agents(walls=[]):
    survivors = []
    positions = []

    num_survivors = random.randint(NUM_SURVIVORS[0], NUM_SURVIVORS[1])

    for i in range(num_survivors):
        attempts = 0
        while attempts < 100:
            x = random.randint(100, WIDTH - 100)
            y = random.randint(100, HEIGHT - 100)

            in_wall = False
            for wall in walls:
                closest_x = max(wall.rect.left, min(x, wall.rect.right))
                closest_y = max(wall.rect.top, min(y, wall.rect.bottom))
                distance_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2
                if distance_sq < 900:
                    in_wall = True
                    break

            if not in_wall and all((x - px) ** 2 + (y - py) ** 2 > 22500 for px, py in positions):
                positions.append((x, y))
                survivors.append(Agent(x, y, SURVIVOR_COLOR, False, i))
                break
            attempts += 1

    hunter = None
    if ENABLE_HUNTER:
        hunter_x, hunter_y = WIDTH // 2, HEIGHT // 2
        for wall in walls:
            closest_x = max(wall.rect.left, min(hunter_x, wall.rect.right))
            closest_y = max(wall.rect.top, min(hunter_y, wall.rect.bottom))
            distance_sq = (hunter_x - closest_x) ** 2 + (hunter_y - closest_y) ** 2
            if distance_sq < 900:
                hunter_x = random.randint(100, WIDTH - 100)
                hunter_y = random.randint(100, HEIGHT - 100)
                break
        hunter = Agent(hunter_x, hunter_y, HUNTER_COLOR, True)

    return survivors, hunter


def draw_agent(surface, agent):
    if agent.caught:
        pygame.draw.circle(surface, (100, 100, 100), (int(agent.x), int(agent.y)), agent.radius)
        pygame.draw.line(surface, (200, 0, 0),
                         (agent.x - agent.radius // 2, agent.y - agent.radius // 2),
                         (agent.x + agent.radius // 2, agent.y + agent.radius // 2), 3)
        pygame.draw.line(surface, (200, 0, 0),
                         (agent.x + agent.radius // 2, agent.y - agent.radius // 2),
                         (agent.x - agent.radius // 2, agent.y + agent.radius // 2), 3)
    elif agent.escaped:
        pygame.draw.circle(surface, (0, 200, 0), (int(agent.x), int(agent.y)), agent.radius)
        pygame.draw.line(surface, (255, 255, 255),
                         (agent.x - agent.radius // 2, agent.y),
                         (agent.x, agent.y + agent.radius // 2), 3)
        pygame.draw.line(surface, (255, 255, 255),
                         (agent.x, agent.y + agent.radius // 2),
                         (agent.x + agent.radius // 2, agent.y - agent.radius // 3), 3)
    else:
        color = (255, 255, 0) if agent.fixing_generator else agent.color
        pygame.draw.circle(surface, color, (int(agent.x), int(agent.y)), agent.radius)

        eye_offset = 4
        eye_radius = 3

        look_dx = np.cos(np.radians(agent.vision_direction))
        look_dy = np.sin(np.radians(agent.vision_direction))

        left_eye_x = agent.x - eye_offset + look_dx * 3
        left_eye_y = agent.y - eye_offset + look_dy * 3
        right_eye_x = agent.x + eye_offset + look_dx * 3
        right_eye_y = agent.y - eye_offset + look_dy * 3

        pygame.draw.circle(surface, (255, 255, 255), (int(left_eye_x), int(left_eye_y)), eye_radius)
        pygame.draw.circle(surface, (255, 255, 255), (int(right_eye_x), int(right_eye_y)), eye_radius)

        pupil_offset = 1.5
        left_pupil_x = left_eye_x + look_dx * pupil_offset
        left_pupil_y = left_eye_y + look_dy * pupil_offset
        right_pupil_x = right_eye_x + look_dx * pupil_offset
        right_pupil_y = right_eye_y + look_dy * pupil_offset

        pygame.draw.circle(surface, (0, 0, 0), (int(left_pupil_x), int(left_pupil_y)), eye_radius - 1)
        pygame.draw.circle(surface, (0, 0, 0), (int(right_pupil_x), int(right_pupil_y)), eye_radius - 1)

        if agent.fixing_generator and pygame.time.get_ticks() % 500 < 250:
            pygame.draw.circle(surface, (255, 255, 0), (int(agent.x), int(agent.y + 15)), 5)

        if agent.cooldown > 0:
            pygame.draw.circle(surface, (0, 100, 255), (int(agent.x), int(agent.y + 20)), 3)

    if agent.is_hunter and agent.capture_target:
        distance = np.sqrt((agent.capture_target.x - agent.x) ** 2 + (agent.capture_target.y - agent.y) ** 2)
        if distance < CAPTURE_RADIUS:
            pulse = (np.sin(pygame.time.get_ticks() * 0.01) + 1) * 0.5
            line_width = int(2 + pulse * 2)
            line_color = (255, int(100 + pulse * 100), 50)

            pygame.draw.line(surface, line_color,
                             (int(agent.x), int(agent.y)),
                             (int(agent.capture_target.x), int(agent.capture_target.y)), line_width)

            progress = agent.hold_steps / CAPTURE_HOLD_STEPS
            if progress > 0:
                progress_color = (255, int(100 + pulse * 100), 50, 150)
                progress_surface = pygame.Surface((agent.capture_target.radius * 2 + 10,
                                                   agent.capture_target.radius * 2 + 10),
                                                  pygame.SRCALPHA)

                progress_angle = int(360 * progress)
                pygame.draw.arc(progress_surface, progress_color,
                                (5, 5, agent.capture_target.radius * 2, agent.capture_target.radius * 2),
                                0, np.radians(progress_angle), 4)

                surface.blit(progress_surface,
                             (agent.capture_target.x - agent.capture_target.radius - 5,
                              agent.capture_target.y - agent.capture_target.radius - 5))


def create_transparent_surface(width, height, color):
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill(color)
    return surface


def save_models(survivors, hunter, episode, successful_escapes, epsilon):
    """Сохраняет модели выживших и охотника в разные файлы"""
    try:
        # Сохраняем выживших
        survivors_data = []
        for agent in survivors:
            q_data = {}
            for key, value in agent.q_table.items():
                q_data[key] = value
            survivors_data.append(q_data)

        data_survivors = {
            'survivors_q_tables': survivors_data,
            'episode': episode,
            'successful_escapes': successful_escapes,
            'epsilon': epsilon,
            'timestamp': time.time(),
            'version': 2
        }

        with open(SURVIVORS_SAVE_FILE, 'wb') as f:
            pickle.dump(data_survivors, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Сохраняем охотника, если он есть
        if hunter is not None:
            hunter_data = {}
            for key, value in hunter.q_table.items():
                hunter_data[key] = value

            data_hunter = {
                'hunter_q_table': hunter_data,
                'episode': episode,
                'timestamp': time.time(),
                'version': 2
            }

            with open(HUNTER_SAVE_FILE, 'wb') as f:
                pickle.dump(data_hunter, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_survivors = os.path.getsize(SURVIVORS_SAVE_FILE) // 1024
        file_size_hunter = os.path.getsize(HUNTER_SAVE_FILE) // 1024 if hunter is not None and os.path.exists(
            HUNTER_SAVE_FILE) else 0

        logger.info(f"Сохранено! Эпизод: {episode}, Выжившие: {file_size_survivors}КБ, Охотник: {file_size_hunter}КБ")

    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")


def load_models(survivors, hunter):
    """Загружает модели выживших и охотника из разных файлов"""
    loaded_episode = 0
    loaded_escapes = 0
    loaded_epsilon = 1.0

    # Загружаем выживших
    if os.path.exists(SURVIVORS_SAVE_FILE):
        try:
            with open(SURVIVORS_SAVE_FILE, 'rb') as f:
                data = pickle.load(f)

            if data.get('version', 1) >= 2:
                for i, agent in enumerate(survivors):
                    if i < len(data['survivors_q_tables']):
                        agent.q_table.clear()
                        agent.q_table.update(data['survivors_q_tables'][i])

                loaded_episode = data['episode']
                loaded_escapes = data['successful_escapes']
                loaded_epsilon = data['epsilon']
                logger.info(f"Загружены выжившие! Эпизод: {loaded_episode}, ε: {loaded_epsilon:.3f}")
            else:
                logger.warning("Старая версия формата выживших, требуется переобучение")

        except Exception as e:
            logger.error(f"Ошибка загрузки выживших: {e}")

    # Загружаем охотника, если он есть и файл существует
    if hunter is not None and os.path.exists(HUNTER_SAVE_FILE):
        try:
            with open(HUNTER_SAVE_FILE, 'rb') as f:
                data = pickle.load(f)

            if data.get('version', 1) >= 2:
                hunter.q_table.clear()
                hunter.q_table.update(data['hunter_q_table'])
                logger.info("Загружен охотник!")
            else:
                logger.warning("Старая версия формата охотника, требуется переобучение")

        except Exception as e:
            logger.error(f"Ошибка загрузки охотника: {e}")

    return loaded_episode, loaded_escapes, loaded_epsilon


def reset_episode(survivors, hunter, walls):
    for survivor in survivors:
        attempts = 0
        while attempts < 50:
            x = random.randint(100, WIDTH - 100)
            y = random.randint(100, HEIGHT - 100)

            in_wall = False
            for wall in walls:
                closest_x = max(wall.rect.left, min(x, wall.rect.right))
                closest_y = max(wall.rect.top, min(y, wall.rect.bottom))
                distance_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2
                if distance_sq < 900:
                    in_wall = True
                    break

            if not in_wall:
                survivor.x = x
                survivor.y = y
                survivor.caught = False
                survivor.escaped = False
                survivor.fixing_generator = False
                survivor.cooldown = 0
                survivor.capture_target = None
                survivor.hold_steps = 0
                survivor.last_distance_to_target = float('inf')
                # Сбрасываем систему обнаружения застревания
                survivor.position_history = []
                survivor.stuck_counter = 0
                survivor.consecutive_wall_hits = 0
                survivor.escape_mode = False
                survivor.escape_steps = 0
                survivor.movement_pattern = []
                break
            attempts += 1

    if hunter is not None:
        hunter_x, hunter_y = WIDTH // 2, HEIGHT // 2
        for wall in walls:
            closest_x = max(wall.rect.left, min(hunter_x, wall.rect.right))
            closest_y = max(wall.rect.top, min(hunter_y, wall.rect.bottom))
            distance_sq = (hunter_x - closest_x) ** 2 + (hunter_y - closest_y) ** 2
            if distance_sq < 900:
                hunter_x = random.randint(100, WIDTH - 100)
                hunter_y = random.randint(100, HEIGHT - 100)
                break

        hunter.x = hunter_x
        hunter.y = hunter_y
        hunter.caught = False
        hunter.escaped = False
        hunter.cooldown = 0
        hunter.capture_target = None
        hunter.hold_steps = 0
        hunter.last_distance_to_target = float('inf')
        # Сбрасываем систему обнаружения застревания для охотника
        hunter.position_history = []
        hunter.stuck_counter = 0
        hunter.consecutive_wall_hits = 0
        hunter.escape_mode = False
        hunter.escape_steps = 0
        hunter.movement_pattern = []

    return create_random_generators(None, 150, walls)


# Инициализация игры
logger.info("Инициализация игры...")
walls = create_random_walls()
exits = create_random_exits(None, walls)  # Исправлено: передаем count=None и walls
survivors, hunter = create_agents(walls)
generators = create_random_generators(None, 150, walls)

font = pygame.font.SysFont('arial', 20)
title_font = pygame.font.SysFont('arial', 24, bold=True)

episode, successful_escapes, epsilon = load_models(survivors, hunter)

running = True
paused = False
simulation_speed = DEFAULT_SIMULATION_SPEED
show_vision_cones = True

logger.info(f"Начало обучения с эпизода {episode}")

while running and episode < EPISODES:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
                logger.info("Пауза" if paused else "Продолжение")
            elif event.key == pygame.K_s:
                save_models(survivors, hunter, episode, successful_escapes, epsilon)
            elif event.key == pygame.K_l:
                episode, successful_escapes, epsilon = load_models(survivors, hunter)
            elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                simulation_speed = min(simulation_speed * 2, 10000)
                logger.info(f"Скорость симуляции: {simulation_speed:.0f}x")
            elif event.key == pygame.K_MINUS:
                simulation_speed = max(simulation_speed / 2, 0.125)
                logger.info(f"Скорость симуляции: {simulation_speed:.0f}x")
            elif event.key == pygame.K_0:
                simulation_speed = 1.0
                logger.info("Нормальная скорость")
            elif event.key == pygame.K_v:
                show_vision_cones = not show_vision_cones
                logger.info(f"Конусы зрения: {'вкл' if show_vision_cones else 'выкл'}")

    if not paused:
        generators_fixed = sum(1 for g in generators if g.fixed)

        steps = max(1, int(simulation_speed))
        for _ in range(steps):
            # Создаем список всех активных агентов
            all_agents = survivors
            if hunter is not None:
                all_agents = survivors + [hunter]

            for agent in all_agents:
                if agent.caught or agent.escaped:
                    continue
                agent.update_vision(all_agents, generators, exits, walls)

            for agent in all_agents:
                if agent.caught or agent.escaped:
                    continue

                state = agent.get_optimized_state(all_agents, generators, exits, generators_fixed)
                action = agent.choose_action(state, epsilon)
                reward = agent.smooth_move(action, all_agents, generators, walls)
                next_state = agent.get_optimized_state(all_agents, generators, exits, generators_fixed)
                agent.update_q_value(state, action, reward, next_state)

                if not agent.is_hunter and generators_fixed == len(generators):
                    for exit_pos in exits:
                        if (exit_pos[0] - agent.x) ** 2 + (exit_pos[1] - agent.y) ** 2 < 1225:
                            agent.escaped = True
                            successful_escapes += 1
                            logger.info(f"Выживший {agent.agent_id} сбежал!")

            if all(s.escaped or s.caught for s in survivors):
                episode += 1
                epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

                if episode % PRUNE_EVERY == 0 and episode > 0:
                    logger.info("Очистка Q-table...")
                    total_pruned = 0
                    for agent in all_agents:
                        total_pruned += agent.prune_q_table()
                    logger.info(f"Всего удалено состояний: {total_pruned}")

                if episode % 50 == 0:
                    save_models(survivors, hunter, episode, successful_escapes, epsilon)
                    avg_reward = np.mean(survivors[0].last_rewards) if survivors and survivors[0].last_rewards else 0
                    hunter_avg = np.mean(hunter.last_rewards) if hunter is not None and hunter.last_rewards else 0

                    # Статистика по застреваниям
                    stuck_count = sum(1 for a in survivors if a.escape_mode)
                    q_table_sizes = [len(a.q_table) for a in survivors]
                    if hunter is not None:
                        q_table_sizes.append(len(hunter.q_table))

                    logger.info(
                        f"Эпизод {episode}, ε={epsilon:.3f}, Побед: {successful_escapes}, "
                        f"Застрявших: {stuck_count}, Q-table размеры: {q_table_sizes}")

                generators = reset_episode(survivors, hunter, walls)
                break

    # Отрисовка
    screen.fill(BACKGROUND)

    for wall in walls:
        wall.draw(screen)

    exit_active = generators_fixed == len(generators)
    for exit_pos in exits:
        color = (0, 255, 0) if exit_active else EXIT_COLOR
        size = 18 if exit_active else 15
        pygame.draw.circle(screen, color, exit_pos, size)
        if exit_active and pygame.time.get_ticks() % 1000 < 500:
            pygame.draw.circle(screen, (255, 255, 255), exit_pos, size, 2)

    for generator in generators:
        generator.draw(screen)

    if show_vision_cones:
        for agent in survivors:
            if not agent.caught and not agent.escaped:
                agent.draw_vision_cone(screen)
        if hunter is not None and not hunter.caught and not hunter.escaped:
            hunter.draw_vision_cone(screen)

    for agent in survivors:
        draw_agent(screen, agent)
    if hunter is not None:
        draw_agent(screen, hunter)

    # UI
    ui_left = create_transparent_surface(400, 280, UI_BG)
    screen.blit(ui_left, (10, 10))

    active_survivors = sum(1 for s in survivors if not s.caught and not s.escaped)
    stuck_survivors = sum(1 for s in survivors if s.escape_mode)

    if simulation_speed >= 1:
        speed_display = f"{simulation_speed:.0f}x"
    else:
        speed_display = f"1/{1 / simulation_speed:.0f}x"

    avg_reward = np.mean(survivors[0].last_rewards) if survivors and survivors[0].last_rewards else 0
    hunter_avg = np.mean(hunter.last_rewards) if hunter is not None and hunter.last_rewards else 0

    stats = [
        f"Q-Learning - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ",
        f"Эпизод: {episode}/{EPISODES}",
        f"Побеги: {successful_escapes}",
        f"Генераторы: {generators_fixed}/{len(generators)}",
        f"Активные: {active_survivors}/{len(survivors)}",
        f"Застрявшие: {stuck_survivors}",
        f"Охотник: {'вкл' if hunter is not None else 'выкл'}",
        f"ε: {epsilon:.3f}",
        f"Скорость: {speed_display}",
        f"Награда выжившего: {avg_reward:.2f}",
        f"Награда охотника: {hunter_avg:.2f}",
        f"Q-table размер: {len(survivors[0].q_table) if survivors else 0}"
    ]

    for i, text in enumerate(stats):
        text_color = TEXT_COLOR
        if i == 5 and stuck_survivors > 0:  # Подсвечиваем строку с застрявшими
            text_color = (255, 100, 100)
        text_surface = font.render(text, True, text_color)
        screen.blit(text_surface, (20, 15 + i * 25))

    ui_right = create_transparent_surface(250, 200, UI_BG)
    screen.blit(ui_right, (WIDTH - 260, 10))

    controls = [
        "Управление:",
        "Пробел - пауза",
        "+ - ускорить x2",
        "- - замедлить /2",
        "0 - нормальная скорость",
        "S - сохранить",
        "L - загрузить",
        "V - зрение вкл/выкл"
    ]

    for i, text in enumerate(controls):
        text_surface = font.render(text, True, TEXT_COLOR)
        screen.blit(text_surface, (WIDTH - 250, 20 + i * 25))

    if paused:
        pause_ui = create_transparent_surface(500, 60, UI_BG)
        screen.blit(pause_ui, (WIDTH // 2 - 250, HEIGHT // 2 - 30))
        pause_text = title_font.render("ПАУЗА - Нажмите ПРОБЕЛ для продолжения", True, (255, 50, 50))
        screen.blit(pause_text, (WIDTH // 2 - pause_text.get_width() // 2, HEIGHT // 2 - 10))

    if simulation_speed != 1.0:
        speed_ui = create_transparent_surface(200, 40, UI_BG)
        screen.blit(speed_ui, (WIDTH // 2 - 100, 10))
        speed_text = font.render(f"СКОРОСТЬ: {speed_display}", True, (255, 165, 0))
        screen.blit(speed_text, (WIDTH // 2 - speed_text.get_width() // 2, 20))

    pygame.display.flip()
    clock.tick(FPS)

logger.info("Завершение обучения...")
save_models(survivors, hunter, episode, successful_escapes, epsilon)
pygame.quit()