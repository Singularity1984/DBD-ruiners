import pygame
import numpy as np
import random
import asyncio
from collections import defaultdict
from config import *
from dqn import DQNAgent


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

        # DQN обучение
        state_size = 20 if is_hunter else 25
        action_size = 5
        self.dqn_agent = DQNAgent(state_size, action_size, is_hunter)
        
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
        self.visible_walls = []
        self.last_action = 3
        
        # Vision rays
        self.vision_rays = 8
        self.ray_distances = []

        # Movement tracking
        self.position_history = []
        self.stuck_counter = 0
        self.consecutive_wall_hits = 0
        self.escape_mode = False
        self.escape_steps = 0
        self.movement_pattern = []

    def _check_wall_collision(self, x, y, walls):
        """Оптимизированная проверка коллизий со стенами"""
        agent_rect = pygame.Rect(x - self.radius, y - self.radius,
                                self.radius * 2, self.radius * 2)
        return any(agent_rect.colliderect(wall.rect) for wall in walls)

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

        total_movement = sum(
            np.sqrt((recent[i][0] - recent[i-1][0])**2 + (recent[i][1] - recent[i-1][1])**2)
            for i in range(1, len(recent))
        )
        avg_movement = total_movement / max(1, len(recent) - 1)
        return avg_movement < 1.0 or self.consecutive_wall_hits > 8


class AsyncAgent(BaseAgent):
    """Асинхронный агент с улучшенной логикой"""

    def __init__(self, x, y, color, is_hunter=False, agent_id=0):
        super().__init__(x, y, color, is_hunter, agent_id)

        # Capture system
        self.capture_target = None
        self.hold_steps = 0
        self.last_distance_to_target = float('inf')

        # Для выжившего
        self.fixing_generator = False

    async def update_vision_async(self, other_agents, generators, exits, walls):
        """Оптимизированное обновление зрения"""
        self.visible_generators = []
        self.visible_survivors = []
        self.visible_hunter = None
        self.visible_exits = []
        self.visible_walls = []
        self.ray_distances = []

        # Обновление направления взгляда
        direction_map = {0: 90, 1: 270, 2: 180, 3: 0}
        self.vision_direction = direction_map.get(self.last_action, 0)

        # Создаем лучи зрения
        if self.vision_rays > 0:
            ray_angles = [
                self.vision_direction - self.vision_angle / 2 + 
                (i * self.vision_angle / max(1, self.vision_rays - 1))
                for i in range(self.vision_rays)
            ]

            # Параллельная проверка лучей
            ray_tasks = [
                self._calculate_ray_distance(angle, walls)
                for angle in ray_angles
            ]
            self.ray_distances = await asyncio.gather(*ray_tasks)
        
        # Дополняем до нужного количества
        while len(self.ray_distances) < 8:
            self.ray_distances.append(self.vision_radius)

        # Проверка видимости объектов
        vision_tasks = []
        for gen in generators:
            vision_tasks.append(self.is_object_in_vision_async(gen.x, gen.y, gen.radius, walls))

        for agent in other_agents:
            if agent != self:
                vision_tasks.append(self.is_object_in_vision_async(agent.x, agent.y, agent.radius, walls))

        for exit_pos in exits:
            vision_tasks.append(self.is_object_in_vision_async(exit_pos[0], exit_pos[1], 20, walls))

        for wall in walls:
            vision_tasks.append(self.is_wall_in_vision_async(wall))

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

        for wall in walls:
            if results[idx]:
                self.visible_walls.append(wall)
            idx += 1

    async def _calculate_ray_distance(self, angle, walls):
        """Асинхронный расчет расстояния до препятствия по лучу"""
        rad = np.radians(angle)
        min_distance = self.vision_radius
        
        for wall in walls:
            distance = self._ray_wall_intersection(self.x, self.y, rad, wall)
            if distance is not None and distance < min_distance:
                min_distance = distance
        
        return min_distance

    def _ray_wall_intersection(self, x, y, angle, wall):
        """Оптимизированный ray casting"""
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        
        tmin = 0.0
        tmax = self.vision_radius
        
        if abs(dx) > 1e-6:
            t1 = (wall.rect.left - x) / dx
            t2 = (wall.rect.right - x) / dx
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
        
        if abs(dy) > 1e-6:
            t1 = (wall.rect.top - y) / dy
            t2 = (wall.rect.bottom - y) / dy
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
        
        return tmin if (tmin <= tmax and tmin >= 0) else None

    async def is_object_in_vision_async(self, obj_x, obj_y, obj_radius, walls):
        """Проверка видимости объекта с учетом препятствий"""
        dx = obj_x - self.x
        dy = obj_y - self.y
        distance_sq = dx ** 2 + dy ** 2
        max_distance = self.vision_radius + obj_radius

        if distance_sq > max_distance ** 2:
            return False

        # Проверка на закрытие стеной
        distance = np.sqrt(distance_sq)
        angle_to_obj = np.arctan2(dy, dx)
        
        for wall in walls:
            intersection = self._ray_wall_intersection(self.x, self.y, angle_to_obj, wall)
            if intersection is not None and intersection < distance:
                return False

        # Проверка угла зрения
        angle_to_obj_deg = np.degrees(angle_to_obj)
        angle_to_obj_deg = (angle_to_obj_deg + 360) % 360
        if angle_to_obj_deg > 180:
            angle_to_obj_deg -= 360

        angle_diff = abs(angle_to_obj_deg - self.vision_direction)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff <= self.vision_angle / 2

    async def is_wall_in_vision_async(self, wall):
        """Проверка видимости стены"""
        closest_x = max(wall.rect.left, min(self.x, wall.rect.right))
        closest_y = max(wall.rect.top, min(self.y, wall.rect.bottom))
        
        distance_sq = (closest_x - self.x) ** 2 + (closest_y - self.y) ** 2
        
        if distance_sq > self.vision_radius ** 2:
            return False
            
        angle_to_wall = np.degrees(np.arctan2(closest_y - self.y, closest_x - self.x))
        angle_to_wall = (angle_to_wall + 360) % 360
        if angle_to_wall > 180:
            angle_to_wall -= 360

        angle_diff = abs(angle_to_wall - self.vision_direction)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff <= self.vision_angle / 2

    async def get_state_vector(self, other_agents, generators, exits, generators_fixed, walls):
        """Получение вектора состояния для DQN"""
        await self.update_vision_async(other_agents, generators, exits, walls)

        state = []

        if self.is_hunter:
            # Состояние для охотника (20 параметров)
            active_survivors = [a for a in other_agents if not a.is_hunter and not a.escaped and not a.caught]
            
            state.append(self.x / WIDTH)
            state.append(self.y / HEIGHT)
            
            if active_survivors:
                nearest = min(active_survivors, key=lambda a: (a.x - self.x) ** 2 + (a.y - self.y) ** 2)
                dx = (nearest.x - self.x) / WIDTH
                dy = (nearest.y - self.y) / HEIGHT
                distance = np.sqrt((nearest.x - self.x) ** 2 + (nearest.y - self.y) ** 2) / np.sqrt(WIDTH**2 + HEIGHT**2)
                state.extend([dx, dy, distance])
            else:
                state.extend([0, 0, 1.0])
            
            state.append(min(1.0, len(self.visible_survivors) / 4.0))
            
            # Лучи (8)
            for dist in self.ray_distances[:8]:
                state.append(min(1.0, dist / self.vision_radius))
            
            state.append(1.0 if self.capture_target else 0.0)
            state.append(min(1.0, self.hold_steps / CAPTURE_HOLD_STEPS))
            state.append(1.0 if self.escape_mode else 0.0)
            
            # Дополняем до 20
            while len(state) < 20:
                state.append(0.0)
            state = state[:20]
            
        else:
            # Состояние для выжившего (25 параметров)
            state.append(self.x / WIDTH)
            state.append(self.y / HEIGHT)
            
            hunter = next((a for a in other_agents if a.is_hunter), None)
            if hunter:
                dx = (hunter.x - self.x) / WIDTH
                dy = (hunter.y - self.y) / HEIGHT
                distance = np.sqrt((hunter.x - self.x) ** 2 + (hunter.y - self.y) ** 2) / np.sqrt(WIDTH**2 + HEIGHT**2)
                state.extend([dx, dy, distance, 1.0 if self.visible_hunter else 0.0])
            else:
                state.extend([0, 0, 1.0, 0.0])
            
            if self.visible_generators:
                nearest_gen = min(self.visible_generators, key=lambda g: (g.x - self.x) ** 2 + (g.y - self.y) ** 2)
                dx = (nearest_gen.x - self.x) / WIDTH
                dy = (nearest_gen.y - self.y) / HEIGHT
                distance = np.sqrt((nearest_gen.x - self.x) ** 2 + (nearest_gen.y - self.y) ** 2) / np.sqrt(WIDTH**2 + HEIGHT**2)
                state.extend([dx, dy, distance, 1.0 if not nearest_gen.fixed else 0.0])
            else:
                state.extend([0, 0, 1.0, 0.0])
            
            if self.visible_exits and generators_fixed == len(generators):
                nearest_exit = min(self.visible_exits, key=lambda e: (e[0] - self.x) ** 2 + (e[1] - self.y) ** 2)
                dx = (nearest_exit[0] - self.x) / WIDTH
                dy = (nearest_exit[1] - self.y) / HEIGHT
                distance = np.sqrt((nearest_exit[0] - self.x) ** 2 + (nearest_exit[1] - self.y) ** 2) / np.sqrt(WIDTH**2 + HEIGHT**2)
                state.extend([dx, dy, distance])
            else:
                state.extend([0, 0, 1.0])
            
            state.append(generators_fixed / len(generators) if generators else 0.0)
            
            # Лучи (8)
            for dist in self.ray_distances[:8]:
                state.append(min(1.0, dist / self.vision_radius))
            
            if hunter and hunter.capture_target == self:
                state.extend([1.0, min(1.0, hunter.hold_steps / CAPTURE_HOLD_STEPS)])
            else:
                state.extend([0.0, 0.0])
            
            state.append(1.0 if self.escape_mode else 0.0)
            
            # Дополняем до 25
            while len(state) < 25:
                state.append(0.0)
            state = state[:25]

        return np.array(state, dtype=np.float32)

    async def smooth_move_async(self, action, agents, generators, walls):
        """Оптимизированное движение"""
        if self.caught or self.escaped:
            return 0, False

        old_x, old_y = self.x, self.y
        reward = -0.01  # Меньший базовый штраф

        current_speed = self.speed * (0.7 if not self.is_hunter and self.cooldown > 0 else 1.0)
        if not self.is_hunter:
            self.cooldown = max(0, self.cooldown - 1)

        # Движение
        direction_map = {
            0: (0, -current_speed),   # up
            1: (0, current_speed),    # down
            2: (-current_speed, 0),    # left
            3: (current_speed, 0)     # right
        }
        
        dx, dy = direction_map.get(action, (0, 0))
        new_x = np.clip(self.x + dx, self.radius, WIDTH - self.radius)
        new_y = np.clip(self.y + dy, self.radius, HEIGHT - self.radius)

        # Проверка коллизий
        x_collision = self._check_wall_collision(new_x, self.y, walls)
        y_collision = self._check_wall_collision(self.x, new_y, walls)

        # Избегание стен на основе зрения
        if self.visible_walls and not self.escape_mode:
            for wall in self.visible_walls[:3]:  # Ограничиваем проверку
                closest_x = max(wall.rect.left, min(self.x, wall.rect.right))
                closest_y = max(wall.rect.top, min(self.y, wall.rect.bottom))
                distance = np.sqrt((closest_x - self.x) ** 2 + (closest_y - self.y) ** 2)
                
                if distance < 30:
                    wall_vec = np.array([closest_x - self.x, closest_y - self.y])
                    move_vec = np.array([dx, dy])
                    if np.linalg.norm(wall_vec) > 0 and np.linalg.norm(move_vec) > 0:
                        wall_vec_norm = wall_vec / np.linalg.norm(wall_vec)
                        move_vec_norm = move_vec / np.linalg.norm(move_vec)
                        dot = np.dot(move_vec_norm, wall_vec_norm)
                        
                        if dot > 0.5:
                            reward -= 1.0
                            if abs(dx) > abs(dy):
                                new_x = self.x
                            else:
                                new_y = self.y

        if not x_collision:
            self.x = new_x
            self.consecutive_wall_hits = 0
        else:
            reward -= 0.5
            self.consecutive_wall_hits += 1

        if not y_collision:
            self.y = new_y
            self.consecutive_wall_hits = 0
        else:
            reward -= 0.5
            self.consecutive_wall_hits += 1

        # Обновление истории
        self.position_history.append((self.x, self.y))
        if len(self.position_history) > POSITION_MEMORY:
            self.position_history.pop(0)

        if len(self.position_history) >= 2:
            dx_hist = self.position_history[-1][0] - self.position_history[-2][0]
            dy_hist = self.position_history[-1][1] - self.position_history[-2][1]
            self.movement_pattern.append((np.sign(dx_hist), np.sign(dy_hist)))
            if len(self.movement_pattern) > 20:
                self.movement_pattern.pop(0)

        # Проверка застревания
        done = False
        if self._is_truly_stuck():
            if not self.escape_mode:
                self.escape_mode = True
                self.escape_steps = 0
                reward -= 5
            else:
                self.escape_steps += 1
                if self.escape_steps > 30:
                    reward -= 2
        else:
            if self.escape_mode:
                self.escape_mode = False
                reward += 10
            self.escape_steps = 0

        await self.update_vision_async(agents, generators, [], walls)

        # Расчет наград
        if self.is_hunter:
            reward = await self._calculate_hunter_reward(agents, reward)
        else:
            reward, done = await self._calculate_survivor_reward(agents, generators, old_x, old_y, action, reward)

        self.last_rewards.append(reward)
        if len(self.last_rewards) > 100:
            self.last_rewards.pop(0)

        return reward, done

    async def _calculate_hunter_reward(self, agents, reward):
        """Расчет награды для охотника"""
        active_survivors = [a for a in agents if not a.is_hunter and not a.escaped and not a.caught]

        if active_survivors:
            nearest = min(active_survivors, key=lambda a: (a.x - self.x) ** 2 + (a.y - self.y) ** 2)
            current_distance = np.sqrt((nearest.x - self.x) ** 2 + (nearest.y - self.y) ** 2)

            if self.last_distance_to_target != float('inf'):
                if current_distance < self.last_distance_to_target:
                    reward += 1.5
                elif current_distance > self.last_distance_to_target + 10:
                    reward -= 1.5

            self.last_distance_to_target = current_distance

            if current_distance < CAPTURE_RADIUS:
                if self.capture_target != nearest:
                    self.capture_target = nearest
                    self.hold_steps = 0
                    reward += 10

                self.hold_steps += 1
                reward += 2

                if self.hold_steps >= CAPTURE_HOLD_STEPS:
                    self.capture_target.caught = True
                    reward += 150
                    self.capture_target = None
                    self.hold_steps = 0
            else:
                if self.capture_target == nearest:
                    reward -= 5
                    self.capture_target = None
                    self.hold_steps = 0

            if self.visible_survivors:
                reward += 0.5
                
            if self.visible_walls:
                reward -= 0.3
        else:
            self.capture_target = None
            self.hold_steps = 0

        return reward

    async def _calculate_survivor_reward(self, agents, generators, old_x, old_y, action, reward):
        """Расчет награды для выжившего"""
        self.fixing_generator = False
        done = False

        hunter = next((a for a in agents if a.is_hunter), None)
        if hunter:
            hunter_distance = np.sqrt((hunter.x - self.x) ** 2 + (hunter.y - self.y) ** 2)
            old_hunter_dist = np.sqrt((hunter.x - old_x) ** 2 + (hunter.y - old_y) ** 2)

            if hunter_distance > old_hunter_dist + 10:
                reward += 3
            elif hunter_distance < old_hunter_dist - 5:
                reward -= 2

            if hunter_distance < CAPTURE_RADIUS:
                reward -= 5
                if hunter.capture_target == self:
                    reward -= 5

            if hunter_distance > CAPTURE_RADIUS and old_hunter_dist < CAPTURE_RADIUS:
                reward += 25

        if self.visible_generators:
            reward += 1

        if self.visible_walls:
            reward -= 0.5

        for generator in self.visible_generators:
            if not generator.fixed:
                distance_sq = (generator.x - self.x) ** 2 + (generator.y - self.y) ** 2
                old_gen_dist_sq = (generator.x - old_x) ** 2 + (generator.y - old_y) ** 2

                if distance_sq < old_gen_dist_sq:
                    reward += 3

                if distance_sq < generator.repair_radius ** 2:
                    if action == 4:
                        self.fixing_generator = True
                        generator.progress += 10
                        if generator.progress >= 100:
                            generator.fixed = True
                            reward += 300
                            self.cooldown = 10
                        else:
                            reward += 15
                    else:
                        reward -= 0.3
                else:
                    if action == 4:
                        reward -= 0.3

        if self.escaped:
            reward += 200
            done = True

        if self.caught:
            reward -= 100
            done = True

        return reward, done
