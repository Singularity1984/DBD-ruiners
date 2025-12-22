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
        # Experience Replay буфер (инициализируем для всех агентов)
        self.experience_buffer = []

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
        self.cookies = 0

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

    def _heuristic_action(self):

        if not self.is_hunter:

            if self.visible_exits:
                nearest_exit = min(self.visible_exits, key=lambda e: (e[0] - self.x) ** 2 + (e[1] - self.y) ** 2)
                dx = nearest_exit[0] - self.x
                dy = nearest_exit[1] - self.y
                if abs(dx) > abs(dy):
                    return 3 if dx > 0 else 2
                else:
                    return 1 if dy > 0 else 0
            

            visible_unfixed_gens = [g for g in self.visible_generators if not g.fixed]
            if visible_unfixed_gens:
                nearest_gen = min(visible_unfixed_gens, key=lambda g: (g.x - self.x) ** 2 + (g.y - self.y) ** 2)
                dx = nearest_gen.x - self.x
                dy = nearest_gen.y - self.y
                dist_sq = dx * dx + dy * dy
                

                if dist_sq <= nearest_gen.repair_radius ** 2:
                    return 4  # чинить
                

                if abs(dx) > abs(dy):
                    return 3 if dx > 0 else 2
                else:
                    return 1 if dy > 0 else 0
            

            if hasattr(self, 'visible_hunter') and self.visible_hunter:
                hunter = self.visible_hunter

                dx = self.x - hunter.x
                dy = self.y - hunter.y
                if abs(dx) > abs(dy):
                    return 3 if dx > 0 else 2
                else:
                    return 1 if dy > 0 else 0
        
        if self.is_hunter and self.visible_survivors:
            target = min(self.visible_survivors, key=lambda a: (a.x - self.x) ** 2 + (a.y - self.y) ** 2)
            dx = target.x - self.x
            dy = target.y - self.y
            if abs(dx) > abs(dy):
                return 3 if dx > 0 else 2
            else:
                return 1 if dy > 0 else 0
        
        return None

    def choose_action(self, state, epsilon, current_episode=0, escape_rate=0.0):
        """Выбор действия с использованием epsilon-greedy стратегии и адаптивной эвристикой"""
        # Adaptive epsilon на основе успешности
        if USE_ADAPTIVE_EPSILON and escape_rate > EPSILON_SUCCESS_THRESHOLD:
            # Если производительность хорошая, уменьшаем epsilon быстрее
            current_epsilon = epsilon * EPSILON_SUCCESS_DECAY
        else:
            current_epsilon = epsilon
            
        if self.escape_mode:
            current_epsilon = min(1.0, current_epsilon + ESCAPE_BOOST)

        # Адаптивное использование эвристики - уменьшается со временем
        use_heuristic = False
        if current_episode < HEURISTIC_MAX_EPISODES:
            # Вероятность использования эвристики уменьшается с эпизодами
            heuristic_prob = HEURISTIC_INITIAL_PROB - (HEURISTIC_INITIAL_PROB - HEURISTIC_FINAL_PROB) * (current_episode / HEURISTIC_MAX_EPISODES)
            use_heuristic = random.uniform(0, 1) < heuristic_prob
        else:
            # После HEURISTIC_MAX_EPISODES используем эвристику редко
            use_heuristic = random.uniform(0, 1) < HEURISTIC_FINAL_PROB

        if use_heuristic:
            heuristic = self._heuristic_action()
            if heuristic is not None:
                action = heuristic
                self.last_action = action
                self.state_visits[state] += 1
                return action

        if random.uniform(0, 1) < current_epsilon:
            action = random.randint(0, 4)
        else:
            action = np.argmax(self.q_table[state])

        self.last_action = action
        self.state_visits[state] += 1
        return action

    def update_q_value(self, state, action, reward, next_state):
        """Обновление Q-values с поддержкой Experience Replay"""
        # Сохраняем опыт в буфер
        if USE_EXPERIENCE_REPLAY:
            # на случай старых объектов из загрузки
            if not hasattr(self, 'experience_buffer'):
                self.experience_buffer = []
            self.experience_buffer.append((state, action, reward, next_state))
            if len(self.experience_buffer) > REPLAY_BUFFER_SIZE:
                self.experience_buffer.pop(0)
        
        # Обычное обновление Q-value
        current_q = self.q_table[state][action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
        self.q_table[state][action] = new_q
        self.last_rewards.append(reward)
        if len(self.last_rewards) > 100:
            self.last_rewards.pop(0)
    
    def replay_experience(self):
        """Переобучение на опыте из буфера"""
        if not USE_EXPERIENCE_REPLAY:
            return
        if not hasattr(self, 'experience_buffer'):
            self.experience_buffer = []
        if len(self.experience_buffer) < REPLAY_BATCH_SIZE:
            return
        
        # Выбираем случайный батч из буфера
        batch = random.sample(self.experience_buffer, min(REPLAY_BATCH_SIZE, len(self.experience_buffer)))
        
        # Переобучаемся на батче
        for state, action, reward, next_state in batch:
            current_q = self.q_table[state][action]
            max_future_q = np.max(self.q_table[next_state])
            new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)
            self.q_table[state][action] = new_q

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

        # Проверяем, все ли генераторы починены (выходы закрыты до починки всех генераторов)
        all_generators_fixed = all(g.fixed for g in generators) if generators else False

        # Обновление направления взгляда с легким сглаживанием
        target_dir = self.vision_direction
        if self.last_action == 0:  # up
            target_dir = 90
        elif self.last_action == 1:  # down
            target_dir = 270
        elif self.last_action == 2:  # left
            target_dir = 180
        elif self.last_action == 3:  # right
            target_dir = 0
        # интерполяция (сглаживание) чтобы убрать "тряску"
        alpha = 0.15
        # кратчайший путь по окружности
        diff = (target_dir - self.vision_direction + 540) % 360 - 180
        self.vision_direction = (self.vision_direction + diff * alpha) % 360

        # Асинхронная проверка объектов
        vision_tasks = []
        for gen in generators:
            vision_tasks.append(self.is_object_in_vision_async(gen.x, gen.y, gen.radius, walls))

        for agent in other_agents:
            if agent != self:
                vision_tasks.append(self.is_object_in_vision_async(agent.x, agent.y, agent.radius, walls))

        # Проверяем видимость выходов только если все генераторы починены
        exit_vision_tasks = []
        if all_generators_fixed:
            for exit_pos in exits:
                exit_vision_tasks.append(self.is_object_in_vision_async(exit_pos[0], exit_pos[1], 20, walls))

        results = await asyncio.gather(*vision_tasks)
        exit_results = await asyncio.gather(*exit_vision_tasks) if exit_vision_tasks else []

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
                        # Не показываем пойманных или сбежавших выживших
                        if not agent.caught and not agent.escaped:
                            self.visible_survivors.append(agent)
                idx += 1

        # Добавляем выходы в видимые только если все генераторы починены
        if all_generators_fixed:
            exit_idx = 0
            for exit_pos in exits:
                if exit_idx < len(exit_results) and exit_results[exit_idx]:
                    self.visible_exits.append(exit_pos)
                exit_idx += 1

    def _is_line_blocked(self, obj_x, obj_y, walls):
        """Проверка пересечения луча со стенами"""
        if not walls:
            return False
        start = (self.x, self.y)
        end = (obj_x, obj_y)
        for wall in walls:
            if wall.rect.clipline(start, end):
                return True
        return False

    async def is_object_in_vision_async(self, obj_x, obj_y, obj_radius=0, walls=None):
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

        if angle_diff > self.vision_angle / 2:
            return False

        if walls and self._is_line_blocked(obj_x, obj_y, walls):
            return False

        return True

    async def get_optimized_state_async(self, other_agents, generators, exits, generators_fixed, walls):
        """Асинхронное получение оптимизированного состояния"""
        await self.update_vision_async(other_agents, generators, exits, walls)

        stuck_indicator = 1 if self.escape_mode else 0

        if self.is_hunter:
            return await self._get_hunter_state(other_agents, stuck_indicator)
        else:
            return await self._get_survivor_state(other_agents, generators, exits, generators_fixed, stuck_indicator)

    async def _get_hunter_state(self, other_agents, stuck_indicator):
        """Улучшенное состояние для охотника с расширенной информацией"""
        active_survivors = [a for a in other_agents if not a.is_hunter and not a.escaped and not a.caught]

        if not active_survivors:
            return (4, 0, 0, stuck_indicator)  # очень далеко, нет направления

        nearest = min(active_survivors, key=lambda a: (a.x - self.x) ** 2 + (a.y - self.y) ** 2)
        distance = np.sqrt((nearest.x - self.x) ** 2 + (nearest.y - self.y) ** 2)
        distance_state = self._get_distance_state(distance)
        direction = self._get_direction_to(nearest.x, nearest.y)

        if self.capture_target and self.capture_target in active_survivors:
            capture_dist = np.sqrt((self.capture_target.x - self.x) ** 2 + (self.capture_target.y - self.y) ** 2)
            if capture_dist < CAPTURE_RADIUS:
                capture_progress = min(2, int(self.hold_steps / (CAPTURE_HOLD_STEPS / 3)))
                return (distance_state, direction, capture_progress, stuck_indicator)

        return (distance_state, direction, 0, stuck_indicator)

    def _get_direction_to(self, target_x, target_y):
        """Получить направление к цели (0-7: 8 направлений)"""
        dx = target_x - self.x
        dy = target_y - self.y
        angle = np.degrees(np.arctan2(dy, dx))
        angle = (angle + 360) % 360
        # Разделяем на 8 направлений
        direction = int(angle / 45) % 8
        return direction

    def _get_distance_state(self, distance):
        """Дискретизация расстояния на состояния (0-4: очень близко -> очень далеко)"""
        if distance < 50:
            return 0
        elif distance < 150:
            return 1
        elif distance < 300:
            return 2
        elif distance < 500:
            return 3
        else:
            return 4

    async def _get_survivor_state(self, other_agents, generators, exits, generators_fixed, stuck_indicator):
        """Улучшенное состояние для выжившего с расширенной информацией"""
        hunter_visible = 1 if self.visible_hunter else 0

        # Получаем информацию о маньяке
        hunter = next((a for a in other_agents if a.is_hunter), None)
        hunter_distance_state = 4  # очень далеко по умолчанию
        hunter_direction = 0
        
        if hunter:
            hunter_distance = np.sqrt((hunter.x - self.x) ** 2 + (hunter.y - self.y) ** 2)
            hunter_distance_state = self._get_distance_state(hunter_distance)
            hunter_direction = self._get_direction_to(hunter.x, hunter.y)
            
            # Проверка захвата охотником
            if hunter.capture_target == self and hunter_distance < CAPTURE_RADIUS:
                capture_state = min(2, int(hunter.hold_steps / (CAPTURE_HOLD_STEPS / 3)))
                return (hunter_visible, hunter_distance_state, hunter_direction, 1, capture_state, stuck_indicator)

        # Если все генераторы починены и выходы открыты - идем к выходам
        if generators_fixed == len(generators):
            if self.visible_exits:
                nearest_exit = min(self.visible_exits, key=lambda e: (e[0] - self.x) ** 2 + (e[1] - self.y) ** 2)
                exit_distance = np.sqrt((nearest_exit[0] - self.x) ** 2 + (nearest_exit[1] - self.y) ** 2)
                exit_distance_state = self._get_distance_state(exit_distance)
                exit_direction = self._get_direction_to(nearest_exit[0], nearest_exit[1])

                return (hunter_visible, hunter_distance_state, hunter_direction, exit_distance_state, exit_direction, 1, stuck_indicator)
            else:
                # Выходы не видим, но ищем их
                return (hunter_visible, hunter_distance_state, hunter_direction, 4, 0, 1, stuck_indicator)

        # Есть непочиненные генераторы - приоритет им
        visible_unfixed_gens = [g for g in self.visible_generators if not g.fixed]
        if visible_unfixed_gens:
            nearest_gen = min(visible_unfixed_gens, key=lambda g: (g.x - self.x) ** 2 + (g.y - self.y) ** 2)
            gen_distance = np.sqrt((nearest_gen.x - self.x) ** 2 + (nearest_gen.y - self.y) ** 2)
            gen_distance_state = self._get_distance_state(gen_distance)
            gen_direction = self._get_direction_to(nearest_gen.x, nearest_gen.y)

            return (hunter_visible, hunter_distance_state, hunter_direction, gen_distance_state, gen_direction, 0, stuck_indicator)
        else:
            # Есть непочиненные генераторы, но они не видны - ищем их
            unfixed_gens = [g for g in generators if not g.fixed]
            if unfixed_gens:
                nearest_gen = min(unfixed_gens, key=lambda g: (g.x - self.x) ** 2 + (g.y - self.y) ** 2)
                gen_distance = np.sqrt((nearest_gen.x - self.x) ** 2 + (nearest_gen.y - self.y) ** 2)
                gen_distance_state = self._get_distance_state(gen_distance)
                gen_direction = self._get_direction_to(nearest_gen.x, nearest_gen.y)

                return (hunter_visible, hunter_distance_state, hunter_direction, gen_distance_state, gen_direction, 0, stuck_indicator)

        # Если маньяк виден, но генераторов нет - убегаем
        return (hunter_visible, hunter_distance_state, hunter_direction, 4, 0, 2, stuck_indicator)

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

        # Если текущая цель поймана - сбрасываем её
        if self.capture_target and (self.capture_target.caught or self.capture_target.escaped):
            self.capture_target = None
            self.hold_steps = 0
            self.last_distance_to_target = float('inf')

        if active_survivors:
            nearest = min(active_survivors, key=lambda a: (a.x - self.x) ** 2 + (a.y - self.y) ** 2)
            current_distance = np.sqrt((nearest.x - self.x) ** 2 + (nearest.y - self.y) ** 2)

            # Печеньки охотника: штраф за дистанцию, бонус только за убийство
            # nearest уже из active_survivors (не пойманные и не сбежавшие)
            self.cookies -= (current_distance / 100.0) * COOKIE_HUNTER_DISTANCE_DECAY

            if hasattr(self, 'last_distance_to_target') and self.last_distance_to_target != float('inf'):
                if current_distance < self.last_distance_to_target:
                    reward += 2
                    self.cookies += COOKIE_HUNTER_APPROACH_GAIN
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
                    self.cookies += COOKIE_HUNTER_KILL_REWARD
                    self.capture_target = None
                    self.hold_steps = 0
                    self.last_distance_to_target = float('inf')
            else:
                if self.capture_target == nearest:
                    reward -= 6
                    self.capture_target = None
                    self.hold_steps = 0
        else:
            self.capture_target = None
            self.hold_steps = 0
            self.last_distance_to_target = float('inf')

        return reward

    async def _calculate_survivor_reward(self, agents, generators, old_x, old_y, action, reward):
        """Расчет награды для выжившего"""
        self.fixing_generator = False

        # Лёгкая «чуйка» на генератор: микропотери и прирост по близости
        # Только для непочиненных генераторов
        active_gens = [g for g in generators if not g.fixed]
        if active_gens:
            nearest_gen = min(active_gens, key=lambda g: (g.x - self.x) ** 2 + (g.y - self.y) ** 2)
            dist = np.sqrt((nearest_gen.x - self.x) ** 2 + (nearest_gen.y - self.y) ** 2)
            self.cookies = max(0, self.cookies - COOKIE_SURVIVOR_SENSE_DECAY)
            closeness = max(0.0, (COOKIE_SURVIVOR_SENSE_MAX_DIST - dist) / COOKIE_SURVIVOR_SENSE_MAX_DIST)
            self.cookies += closeness * COOKIE_SURVIVOR_SENSE_GAIN

        hunter = next((a for a in agents if a.is_hunter), None)
        if hunter:
            hunter_distance = np.sqrt((hunter.x - self.x) ** 2 + (hunter.y - self.y) ** 2)
            old_hunter_dist = np.sqrt((hunter.x - old_x) ** 2 + (hunter.y - old_y) ** 2)

            # Награды за убегание (но приоритет у генераторов)
            if self.visible_hunter:
                if hunter_distance > old_hunter_dist + 10:
                    reward += 3  # награда за убегание (меньше чем за починку генератора)
                elif hunter_distance < old_hunter_dist - 5:
                    reward -= 2  # штраф за приближение к маньяку (но меньше чем штраф за непочинку генератора)
            else:
                # Если маньяка не видим, но он близко - небольшие награды за удаление
                if hunter_distance > old_hunter_dist + 10:
                    reward += 2
                elif hunter_distance < old_hunter_dist - 5:
                    reward -= 1

            if hunter_distance < CAPTURE_RADIUS:
                reward -= 3
                if hunter.capture_target == self:
                    reward -= 4

            if hunter_distance > CAPTURE_RADIUS and old_hunter_dist < CAPTURE_RADIUS:
                reward += 30

        # Проверяем состояние генераторов
        unfixed_gens = [g for g in generators if not g.fixed]
        all_generators_fixed = len(unfixed_gens) == 0
        
        # Если все генераторы починены - идем к выходам
        if all_generators_fixed:
            # Награды за движение к выходам
            if self.visible_exits:
                nearest_exit = min(self.visible_exits, key=lambda e: (e[0] - self.x) ** 2 + (e[1] - self.y) ** 2)
                distance_to_exit_sq = (nearest_exit[0] - self.x) ** 2 + (nearest_exit[1] - self.y) ** 2
                old_exit_dist_sq = (nearest_exit[0] - old_x) ** 2 + (nearest_exit[1] - old_y) ** 2
                
                if distance_to_exit_sq < old_exit_dist_sq:
                    reward += 8  # награда за приближение к выходу
                elif distance_to_exit_sq > old_exit_dist_sq + 5:
                    reward -= 1  # штраф за удаление от выхода
            else:
                # Ищем выходы (поощряем исследование)
                reward += 0.5
        
        # Всегда хотим чинить генератор (приоритет над убеганием от маньяка)
        # Только для непочиненных генераторов - починенные не дают награды
        visible_unfixed_gens = [g for g in self.visible_generators if not g.fixed]
        if visible_unfixed_gens:
            # Базовая награда за видимость непочиненного генератора (побуждает идти к нему)
            reward += 1
            
        for generator in visible_unfixed_gens:
            # Дополнительная проверка: генератор не должен быть починен
            if generator.fixed:
                continue  # Пропускаем починенные генераторы - не даем награды
            
            distance_sq = (generator.x - self.x) ** 2 + (generator.y - self.y) ** 2
            old_gen_dist_sq = (generator.x - old_x) ** 2 + (generator.y - old_y) ** 2

            # Поощряем движение только к непочиненному генератору (независимо от маньяка)
            if distance_sq < old_gen_dist_sq:
                reward += 6  # увеличенная награда за приближение к генератору
                self.cookies += COOKIE_SURVIVOR_SPOT
            elif distance_sq > old_gen_dist_sq + 5:
                # Штраф за удаление от генератора (если видим его)
                reward -= 1

            # Проверяем еще раз перед починкой (генератор мог быть починен другим выжившим)
            if not generator.fixed and distance_sq < generator.repair_radius ** 2:
                if action == 4:  # Действие починки
                    # Поощряем починку генератора (высокий приоритет)
                    self.fixing_generator = True
                    self.cookies += COOKIE_SURVIVOR_FIXING
                    generator.progress += 10
                    if generator.progress >= 100:
                        generator.fixed = True
                        reward += 250  # большая награда за завершение генератора
                        self.cooldown = 10
                        self.cookies += COOKIE_SURVIVOR_FINISH
                    else:
                        reward += 15  # увеличенная награда за починку (больше чем за убегание)
                else:
                    # Штраф за то, что генератор рядом, но не чиним
                    reward -= 3  # увеличенный штраф за игнорирование генератора
            else:
                if action == 4:
                    reward -= 0.5  # штраф за попытку чинить издалека
        
        # Если есть непочиненные генераторы, но они не видны - ищем их
        if unfixed_gens and not visible_unfixed_gens:
            nearest_unfixed_gen = min(unfixed_gens, key=lambda g: (g.x - self.x) ** 2 + (g.y - self.y) ** 2)
            distance_to_gen_sq = (nearest_unfixed_gen.x - self.x) ** 2 + (nearest_unfixed_gen.y - self.y) ** 2
            old_gen_dist_sq = (nearest_unfixed_gen.x - old_x) ** 2 + (nearest_unfixed_gen.y - old_y) ** 2
            
            if distance_to_gen_sq < old_gen_dist_sq:
                reward += 3  # награда за приближение к невидимому генератору
            elif distance_to_gen_sq > old_gen_dist_sq + 10:
                reward -= 0.5  # небольшой штраф за удаление

        if self.escaped:
            reward += 150

        if self.caught:
            reward -= 80

        return reward