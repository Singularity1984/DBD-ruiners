import pygame
import numpy as np
import random
import asyncio
from collections import defaultdict
from config import *
from q_learning import SmartQTable


class BaseAgent:
    """Базовый класс для всех агентов"""

    def __init__(self, x, y, color, is_hunter=False, agent_id=0):
        self.x = x
        self.y = y
        self.color = color
        self.speed = HUNTER_SPEED if is_hunter else SURVIVOR_SPEED
        self.radius = HUNTER_RADIUS if is_hunter else SURVIVOR_RADIUS
        self.is_hunter = is_hunter
        self.agent_id = agent_id

        # Q-learning
        self.q_table = SmartQTable(MAX_Q_TABLE_SIZE)
        self.epsilon = 1.0
        self.caught = False
        self.escaped = False
        self.cooldown = 0
        self.last_rewards = []

        # Vision system
        self.vision_direction = 0
        self.vision_radius = HUNTER_VISION_RADIUS if is_hunter else SURVIVOR_VISION_RADIUS
        self.vision_angle = HUNTER_VISION_ANGLE if is_hunter else SURVIVOR_VISION_ANGLE
        self.visible_generators = []
        self.visible_survivors = []
        self.visible_hunter = None
        self.visible_exits = []
        self.last_action = 3

        # Movement tracking
        self.position_history = []
        self.stuck_counter = 0
        self.consecutive_wall_hits = 0
        self.escape_mode = False
        self.escape_steps = 0
        self.movement_pattern = []

        # Statistics
        self.state_visits = defaultdict(int)

    def _check_wall_collision(self, x, y, walls):
        """Проверка коллизий со стенами"""
        new_rect = pygame.Rect(x - self.radius, y - self.radius,
                               self.radius * 2, self.radius * 2)

        for wall in walls:
            if new_rect.colliderect(wall.rect):
                return True
        return False

    def _is_truly_stuck(self):
        """Определение застревания агента"""
        if len(self.position_history) < STUCK_THRESHOLD:
            return False

        recent = self.position_history[-STUCK_THRESHOLD:]
        unique_positions = len(set((int(x / 10), int(y / 10)) for x, y in recent))

        if unique_positions < 5:
            return True

        if len(self.movement_pattern) >= 10:
            if len(set(self.movement_pattern)) < 4:
                return True

        total_movement = 0
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i - 1][0]
            dy = recent[i][1] - recent[i - 1][1]
            total_movement += np.sqrt(dx ** 2 + dy ** 2)

        avg_movement = total_movement / (len(recent) - 1)
        return avg_movement < 1.0 or self.consecutive_wall_hits > 8

    def choose_action(self, state, epsilon):
        """Выбор действия с использованием epsilon-greedy стратегии"""
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
        """Обновление Q-values"""
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
        self.q_table[state][action] = new_q
        self.last_rewards.append(reward)
        if len(self.last_rewards) > 100:
            self.last_rewards.pop(0)

    def prune_q_table(self, min_visits=3):
        """Очистка редко используемых состояний"""
        if len(self.q_table) < MAX_Q_TABLE_SIZE * 0.7:
            return 0

        removed = self.q_table.prune_infrequent(min_visits)

        keys_to_remove = [state for state in self.state_visits
                          if self.state_visits[state] < min_visits]
        for state in keys_to_remove:
            del self.state_visits[state]

        return removed


class AsyncAgent(BaseAgent):
    """Асинхронный агент с улучшенной логикой"""

    def __init__(self, x, y, color, is_hunter=False, agent_id=0):
        super().__init__(x, y, color, is_hunter, agent_id)

        # Capture system (для охотника)
        self.capture_target = None
        self.capture_progress = 0
        self.hold_steps = 0
        self.last_distance_to_target = float('inf')

        # Для выжившего
        self.fixing_generator = False

    async def update_vision_async(self, other_agents, generators, exits, walls):
        """Асинхронное обновление зрения"""
        self.visible_generators = []
        self.visible_survivors = []
        self.visible_hunter = None
        self.visible_exits = []

        # Обновление направления взгляда
        if self.last_action == 0:  # up
            self.vision_direction = 90
        elif self.last_action == 1:  # down
            self.vision_direction = 270
        elif self.last_action == 2:  # left
            self.vision_direction = 180
        elif self.last_action == 3:  # right
            self.vision_direction = 0

        # Асинхронная проверка объектов
        vision_tasks = []
        for gen in generators:
            vision_tasks.append(self.is_object_in_vision_async(gen.x, gen.y, gen.radius))

        for agent in other_agents:
            if agent != self:
                vision_tasks.append(self.is_object_in_vision_async(agent.x, agent.y, agent.radius))

        for exit_pos in exits:
            vision_tasks.append(self.is_object_in_vision_async(exit_pos[0], exit_pos[1], 20))

        results = await asyncio.gather(*vision_tasks)

        # Обработка результатов
        idx = 0
        for gen in generators:
            if results[idx]:
                self.visible_generators.append(gen)
            idx += 1

        for agent in other_agents:
            if agent != self:
                if results[idx]:
                    if agent.is_hunter:
                        self.visible_hunter = agent
                    else:
                        self.visible_survivors.append(agent)
                idx += 1

        for exit_pos in exits:
            if results[idx]:
                self.visible_exits.append(exit_pos)
            idx += 1

    async def is_object_in_vision_async(self, obj_x, obj_y, obj_radius=0):
        """Асинхронная проверка видимости объекта"""
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

    async def get_optimized_state_async(self, other_agents, generators, exits, generators_fixed):
        """Асинхронное получение оптимизированного состояния"""
        await self.update_vision_async(other_agents, generators, exits, [])

        stuck_indicator = 1 if self.escape_mode else 0

        if self.is_hunter:
            return await self._get_hunter_state(other_agents, stuck_indicator)
        else:
            return await self._get_survivor_state(other_agents, generators, exits, generators_fixed, stuck_indicator)

    async def _get_hunter_state(self, other_agents, stuck_indicator):
        """Состояние для охотника"""
        active_survivors = [a for a in other_agents if not a.is_hunter and not a.escaped and not a.caught]

        if not active_survivors:
            return (2, 2, 0, stuck_indicator)

        nearest = min(active_survivors, key=lambda a: (a.x - self.x) ** 2 + (a.y - self.y) ** 2)
        distance = np.sqrt((nearest.x - self.x) ** 2 + (nearest.y - self.y) ** 2)

        if distance < 100:
            dist_state = 0
        elif distance < 300:
            dist_state = 1
        else:
            dist_state = 2

        dx = 1 if nearest.x > self.x else 0
        dy = 1 if nearest.y > self.y else 0

        if self.capture_target and self.capture_target in active_survivors:
            capture_dist = np.sqrt((self.capture_target.x - self.x) ** 2 + (self.capture_target.y - self.y) ** 2)
            if capture_dist < CAPTURE_RADIUS:
                capture_progress = min(2, int(self.hold_steps / (CAPTURE_HOLD_STEPS / 3)))
                return (dx, dy, capture_progress, stuck_indicator)

        return (dx, dy, dist_state, stuck_indicator)

    async def _get_survivor_state(self, other_agents, generators, exits, generators_fixed, stuck_indicator):
        """Состояние для выжившего"""
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
                gen_state = 0
            elif distance_to_gen < 250:
                gen_state = 1
            else:
                gen_state = 2

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
            return (hunter_visible, 2, 2, stuck_indicator)

    async def smooth_move_async(self, action, agents, generators, walls):
        """Асинхронное улучшенное движение"""
        if self.caught or self.escaped:
            return 0

        old_x, old_y = self.x, self.y
        reward = -0.05

        current_speed = self.speed
        if not self.is_hunter and self.cooldown > 0:
            current_speed *= 0.7
            self.cooldown -= 1

        # Расчет новой позиции
        new_x, new_y = self.x, self.y
        if action == 0:  # up
            new_y -= current_speed
        elif action == 1:  # down
            new_y += current_speed
        elif action == 2:  # left
            new_x -= current_speed
        elif action == 3:  # right
            new_x += current_speed

        # Проверка границ
        new_x = max(self.radius, min(WIDTH - self.radius, new_x))
        new_y = max(self.radius, min(HEIGHT - self.radius, new_y))

        # Асинхронная проверка коллизий
        collision_tasks = [
            asyncio.to_thread(self._check_wall_collision, new_x, self.y, walls),
            asyncio.to_thread(self._check_wall_collision, self.x, new_y, walls)
        ]
        x_collision, y_collision = await asyncio.gather(*collision_tasks)

        if not x_collision:
            self.x = new_x
            self.consecutive_wall_hits = 0
        else:
            reward -= 0.3
            self.consecutive_wall_hits += 1

        if not y_collision:
            self.y = new_y
            self.consecutive_wall_hits = 0
        else:
            reward -= 0.3
            self.consecutive_wall_hits += 1

        # Обновление истории позиций
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > POSITION_MEMORY:
            self.position_history.pop(0)

        if len(self.position_history) >= 2:
            dx = self.position_history[-1][0] - self.position_history[-2][0]
            dy = self.position_history[-1][1] - self.position_history[-2][1]
            movement = (np.sign(dx), np.sign(dy))
            self.movement_pattern.append(movement)
            if len(self.movement_pattern) > 20:
                self.movement_pattern.pop(0)

        # Проверка застревания
        if self._is_truly_stuck():
            if not self.escape_mode:
                self.escape_mode = True
                self.escape_steps = 0
                reward -= 8
            else:
                self.escape_steps += 1
                if self.escape_steps > 30:
                    reward -= 3
        else:
            if self.escape_mode:
                self.escape_mode = False
                reward += 15
            self.escape_steps = 0

        await self.update_vision_async(agents, generators, [], walls)

        # Логика наград в зависимости от типа агента
        if self.is_hunter:
            reward = await self._calculate_hunter_reward(agents, reward)
        else:
            reward = await self._calculate_survivor_reward(agents, generators, old_x, old_y, action, reward)

        return reward

    async def _calculate_hunter_reward(self, agents, reward):
        """Расчет награды для охотника"""
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

        return reward

    async def _calculate_survivor_reward(self, agents, generators, old_x, old_y, action, reward):
        """Расчет награды для выжившего"""
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