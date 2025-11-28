import asyncio
import random
import numpy as np
import pygame
from collections import defaultdict
from config import *  # ДОБАВЛЕНО: импорт констант напрямую
from QTable import AsyncQTable


class Agent:
    def __init__(self, x, y, color, is_hunter=False, agent_id=0, loaded_q_table=None):
        self.x = x
        self.y = y
        self.color = color
        self.speed = HUNTER_SPEED if is_hunter else SURVIVOR_SPEED
        self.radius = HUNTER_RADIUS if is_hunter else SURVIVOR_RADIUS
        self.is_hunter = is_hunter
        self.agent_id = agent_id

        self.q_table = AsyncQTable(MAX_Q_TABLE_SIZE)
        if loaded_q_table is not None:
            self.q_table._data = loaded_q_table.copy()
        self.epsilon = 1.0
        self.caught = False
        self.escaped = False
        self.fixing_generator = False
        self.cooldown = 0
        self.last_rewards = []

        self.vision_direction = 0
        self.vision_radius = HUNTER_VISION_RADIUS if is_hunter else SURVIVOR_VISION_RADIUS
        self.vision_angle = HUNTER_VISION_ANGLE if is_hunter else SURVIVOR_VISION_ANGLE
        self.visible_generators = []
        self.visible_survivors = []
        self.visible_hunter = None
        self.visible_exits = []
        self.visible_hooks = []
        self.visible_lockers = []
        self.visible_totems = []
        self.last_action = 3

        self.capture_target = None
        self.capture_progress = 0
        self.hold_steps = 0
        self.last_distance_to_target = float('inf')
        self.last_distance_to_exit = float('inf')

        self.state_visits = defaultdict(int)
        self.position_history = []
        self.stuck_counter = 0
        self.consecutive_wall_hits = 0
        self.escape_mode = False
        self.escape_steps = 0
        self.movement_pattern = []

        self.health_state = "healthy"
        self.on_hook = False
        self.hook_progress = 0
        self.hook_stage = 1
        self.in_locker = False
        self.being_carried = False
        self.recovery_progress = 0
        self.carrying_survivor = None
        self.ability_cooldown = 0
        self.ability_charged = True

        self._position_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._action_lock = asyncio.Lock()

        self._distance_cache = {}
        self._vision_cache = {}
        self._state_cache = {}
        self.visited_cells = set()
        self.discovered_generators = set()
        self.discovered_exits = set()

        self.survival_steps = 0

    async def update_vision(self, other_agents, generators, exits, hooks, lockers, totems, walls):
        async with self._state_lock:
            self.visible_generators = []
            self.visible_survivors = []
            self.visible_hunter = None
            self.visible_exits = []
            self.visible_hooks = []
            self.visible_lockers = []
            self.visible_totems = []

            if self.last_action == 0:
                self.vision_direction = 90
            elif self.last_action == 1:
                self.vision_direction = 270
            elif self.last_action == 2:
                self.vision_direction = 180
            elif self.last_action == 3:
                self.vision_direction = 0

            for obj_list, obj_type in [(generators, 'generator'), (exits, 'exit'),
                                       (hooks, 'hook'), (lockers, 'locker'),
                                       (totems, 'totem')]:
                for obj in obj_list:
                    if obj_type == 'exit':
                        obj_x, obj_y = obj
                        obj_radius = 20
                    else:
                        obj_x, obj_y = obj.x, obj.y
                        obj_radius = getattr(obj, 'radius', 20)

                    if await self.is_object_in_vision(obj_x, obj_y, obj_radius):
                        if obj_type == 'generator':
                            self.visible_generators.append(obj)
                        elif obj_type == 'exit':
                            self.visible_exits.append(obj)
                        elif obj_type == 'hook':
                            self.visible_hooks.append(obj)
                        elif obj_type == 'locker':
                            self.visible_lockers.append(obj)
                        elif obj_type == 'totem':
                            self.visible_totems.append(obj)

            for agent in other_agents:
                if agent != self:
                    if await self.is_object_in_vision(agent.x, agent.y, agent.radius):
                        if agent.is_hunter:
                            self.visible_hunter = agent
                        else:
                            self.visible_survivors.append(agent)

    async def is_object_in_vision(self, obj_x, obj_y, obj_radius=0):
        cache_key = (obj_x, obj_y, obj_radius, self.vision_direction)
        if cache_key in self._vision_cache:
            return self._vision_cache[cache_key]

        dx = obj_x - self.x
        dy = obj_y - self.y
        distance = np.sqrt(dx ** 2 + dy ** 2)

        if distance > self.vision_radius + obj_radius:
            self._vision_cache[cache_key] = False
            return False

        angle_to_obj = np.degrees(np.arctan2(dy, dx))
        angle_to_obj = (angle_to_obj + 360) % 360
        if angle_to_obj > 180:
            angle_to_obj -= 360

        angle_diff = abs(angle_to_obj - self.vision_direction)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        result = angle_diff <= self.vision_angle / 2
        self._vision_cache[cache_key] = result
        return result

    async def get_state(self, other_agents, generators, exits, hooks, lockers, totems, generators_fixed):
        await self.update_vision(other_agents, generators, exits, hooks, lockers, totems, [])

        stuck_indicator = 1 if self.escape_mode else 0
        health_state = 0 if self.health_state == "healthy" else 1 if self.health_state == "injured" else 2

        if self.is_hunter:
            active_survivors = [a for a in other_agents if
                                not a.is_hunter and not a.escaped and not a.caught and not a.on_hook]
            if not active_survivors:
                return (2, 2, 0, health_state, stuck_indicator)

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
                    return (dx, dy, capture_progress, health_state, stuck_indicator)

            return (dx, dy, dist_state, health_state, stuck_indicator)
        else:
            hunter_visible = 1 if self.visible_hunter else 0
            hunter = next((a for a in other_agents if a.is_hunter), None)

            if hunter and hunter.capture_target == self:
                distance = np.sqrt((hunter.x - self.x) ** 2 + (hunter.y - self.y) ** 2)
                if distance < CAPTURE_RADIUS:
                    capture_state = min(2, int(hunter.hold_steps / (CAPTURE_HOLD_STEPS / 3)))
                    return (hunter_visible, 1, capture_state, health_state, stuck_indicator)

            # ИСПРАВЛЕНИЕ: учитываем только непочиненные генераторы
            unfixed_generators = [g for g in generators if not g.fixed]
            if unfixed_generators and self.visible_generators:
                # Фильтруем только непочиненные видимые генераторы
                unfixed_visible = [g for g in self.visible_generators if not g.fixed]
                if unfixed_visible:
                    nearest_gen = min(unfixed_visible, key=lambda g: (g.x - self.x) ** 2 + (g.y - self.y) ** 2)
                    distance_to_gen = np.sqrt((nearest_gen.x - self.x) ** 2 + (nearest_gen.y - self.y) ** 2)

                    if distance_to_gen < 100:
                        gen_state = 0
                    elif distance_to_gen < 250:
                        gen_state = 1
                    else:
                        gen_state = 2

                    return (hunter_visible, gen_state, 0, health_state, stuck_indicator)

            # УЛУЧШЕНИЕ: если генераторы починены, но выходы не видны, все равно даем информацию о выходе
            if generators_fixed == len(generators):
                if self.visible_exits:
                    nearest_exit = min(self.visible_exits, key=lambda e: (e[0] - self.x) ** 2 + (e[1] - self.y) ** 2)
                    distance_to_exit = np.sqrt((nearest_exit[0] - self.x) ** 2 + (nearest_exit[1] - self.y) ** 2)

                    if distance_to_exit < 100:
                        exit_state = 0
                    elif distance_to_exit < 250:
                        exit_state = 1
                    else:
                        exit_state = 2

                    return (hunter_visible, exit_state, 1, health_state, stuck_indicator)
                else:
                    # Если выходы не видны, но генераторы починены - находим ближайший выход
                    nearest_exit = min(exits, key=lambda e: (e[0] - self.x) ** 2 + (e[1] - self.y) ** 2)
                    distance_to_exit = np.sqrt((nearest_exit[0] - self.x) ** 2 + (nearest_exit[1] - self.y) ** 2)

                    if distance_to_exit < 300:
                        exit_state = 0
                    elif distance_to_exit < 600:
                        exit_state = 1
                    else:
                        exit_state = 2

                    return (hunter_visible, exit_state, 1, health_state, stuck_indicator)

            # Если нет видимых целей
            return (hunter_visible, 2, 2, health_state, stuck_indicator)

    async def choose_action(self, state, epsilon):
        async with self._action_lock:
            current_epsilon = min(1.0, epsilon + (ESCAPE_BOOST if self.escape_mode else 0))

            if random.uniform(0, 1) < current_epsilon:
                if self.is_hunter:
                    action = random.randint(0, 4)
                else:
                    action = random.randint(0, 11)
            else:
                q_values = await self.q_table.get(state)
                action = np.argmax(q_values)

            self.last_action = action
            self.state_visits[state] += 1
            return action

    async def update_q(self, state, action, reward, next_state):
        current_q = (await self.q_table.get(state))[action]
        max_future_q = np.max(await self.q_table.get(next_state))
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - current_q)

        new_q_values = await self.q_table.get(state)
        new_q_values[action] = new_q
        await self.q_table.set(state, new_q_values)

        self.last_rewards.append(reward)
        if len(self.last_rewards) > 100:
            self.last_rewards.pop(0)

    async def check_wall_collision(self, x, y, walls):
        new_rect = pygame.Rect(x - self.radius, y - self.radius, self.radius * 2, self.radius * 2)
        for wall in walls:
            if new_rect.colliderect(wall.rect):
                return True
        return False

    async def is_stuck(self):
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

    async def move(self, action, agents, generators, exits, hooks, lockers, totems, walls):
        if self.caught or self.escaped or self.on_hook or (self.being_carried and not self.is_hunter):
            return 0

        async with self._position_lock:
            old_x, old_y = self.x, self.y
            reward = -0.01  # Базовая награда за шаг

            if self.ability_cooldown > 0:
                self.ability_cooldown -= 1
                if self.ability_cooldown == 0:
                    self.ability_charged = True

            current_speed = self.speed
            if not self.is_hunter and self.cooldown > 0:
                current_speed *= 0.7
                self.cooldown -= 1

            if self.is_hunter and self.carrying_survivor:
                current_speed *= 0.6

            if action in [0, 1, 2, 3] and not self.in_locker:
                new_x, new_y = self.x, self.y

                if action == 0:
                    new_y -= current_speed
                elif action == 1:
                    new_y += current_speed
                elif action == 2:
                    new_x -= current_speed
                elif action == 3:
                    new_x += current_speed

                new_x = max(self.radius, min(WIDTH - self.radius, new_x))
                new_y = max(self.radius, min(HEIGHT - self.radius, new_y))

                temp_x, temp_y = new_x, self.y
                x_collision = await self.check_wall_collision(temp_x, temp_y, walls)
                if not x_collision:
                    self.x = temp_x
                    self.consecutive_wall_hits = 0
                else:
                    reward -= 0.3
                    self.consecutive_wall_hits += 1

                temp_x, temp_y = self.x, new_y
                y_collision = await self.check_wall_collision(temp_x, temp_y, walls)
                if not y_collision:
                    self.y = temp_y
                    self.consecutive_wall_hits = 0
                else:
                    reward -= 0.3
                    self.consecutive_wall_hits += 1

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

            is_stuck = await self.is_stuck()
            if is_stuck:
                if not self.escape_mode:
                    self.escape_mode = True
                    self.escape_steps = 0
                    reward -= 6
                else:
                    self.escape_steps += 1
                    if self.escape_steps > 30:
                        reward -= 2
            else:
                if self.escape_mode:
                    self.escape_mode = False
                    reward += 12
                self.escape_steps = 0

        # УЛУЧШЕНИЕ: Награды за приближение к цели для выживших
        if not self.is_hunter:
            generators_fixed = sum(1 for g in generators if g.fixed)

            # Награда за приближение к цели
            if generators_fixed < len(generators):
                # Ищем ближайший непочиненный генератор
                unfixed_generators = [g for g in generators if not g.fixed]
                if unfixed_generators:
                    nearest_gen = min(unfixed_generators, key=lambda g: (g.x - self.x) ** 2 + (g.y - self.y) ** 2)
                    new_dist = np.sqrt((nearest_gen.x - self.x) ** 2 + (nearest_gen.y - self.y) ** 2)
                    if new_dist < self.last_distance_to_target:
                        reward += 0.8  # Награда за приближение к генератору
                    self.last_distance_to_target = new_dist
            else:
                # Все генераторы починены - ищем выход
                nearest_exit = min(exits, key=lambda e: (e[0] - self.x) ** 2 + (e[1] - self.y) ** 2)
                new_dist = np.sqrt((nearest_exit[0] - self.x) ** 2 + (nearest_exit[1] - self.y) ** 2)
                if new_dist < self.last_distance_to_exit:
                    reward += 1.2  # Большая награда за приближение к выходу
                self.last_distance_to_exit = new_dist

            if action == 4 and not self.in_locker:
                self.fixing_generator = False
                for gen in self.visible_generators:
                    if not gen.fixed:
                        dist_sq = (gen.x - self.x) ** 2 + (gen.y - self.y) ** 2
                        if dist_sq < gen.repair_radius ** 2:
                            self.fixing_generator = True
                            repaired = await gen.repair(6)
                            if repaired:
                                reward += 500  # УВЕЛИЧЕНО с 250
                                self.cooldown = 10
                            reward += 25  # УВЕЛИЧЕНО с 15
                            break

            elif action == 5 and self.health_state == "injured" and not self.in_locker:
                self.recovery_progress += 12
                if self.recovery_progress >= 100:
                    self.health_state = "healthy"
                    self.recovery_progress = 0
                    reward += 150  # УВЕЛИЧЕНО с 70
                reward += 15  # УВЕЛИЧЕНО с 8

            elif action == 7 and not self.in_locker:
                for hook in self.visible_hooks:
                    if hook.survivor and hook.survivor != self:
                        dist_sq = (hook.x - self.x) ** 2 + (hook.y - self.y) ** 2
                        if dist_sq < 2500:
                            success = random.random() < (0.85 if hook.stage == 1 else 0.65)
                            if success:
                                await hook.remove_survivor()
                                hook.survivor.health_state = "injured"
                                reward += 300  # УВЕЛИЧЕНО с 140
                            else:
                                reward -= 8
                            break

            elif action == 8 and not self.in_locker:
                for totem in self.visible_totems:
                    dist_sq = (totem.x - self.x) ** 2 + (totem.y - self.y) ** 2
                    if dist_sq < 900:
                        cleansed = await totem.cleanse()
                        if cleansed:
                            reward += 50 if totem.is_hex else 25
                        break

            elif action == 10:
                for locker in self.visible_lockers:
                    dist_sq = (locker.x - self.x) ** 2 + (locker.y - self.y) ** 2
                    if dist_sq < 900:
                        if not self.in_locker:
                            entered = await locker.enter()
                            if entered:
                                self.in_locker = True
                                async with self._position_lock:
                                    self.x, self.y = locker.x, locker.y
                                reward += 8
                        else:
                            await locker.exit()
                            self.in_locker = False
                            reward += 4
                        break

            cell_x, cell_y = int(self.x / 40), int(self.y / 40)
            if (cell_x, cell_y) not in self.visited_cells:
                self.visited_cells.add((cell_x, cell_y))
                reward += 16

            # Награда за выживание каждые 50 шагов
            self.survival_steps += 1
            if self.survival_steps % 50 == 0:
                reward += 5

        else:
            active_survivors = [a for a in agents if
                                not a.is_hunter and not a.escaped and not a.caught and not a.on_hook and not a.being_carried]

            if self.carrying_survivor:
                for hook in hooks:
                    if hook.available:
                        dist_sq = (hook.x - self.x) ** 2 + (hook.y - self.y) ** 2
                        if dist_sq < 900:
                            hooked = await hook.add_survivor(self.carrying_survivor)
                            if hooked:
                                self.carrying_survivor.on_hook = True
                                self.carrying_survivor.being_carried = False
                                self.carrying_survivor = None
                                reward += 90
                                break

            if active_survivors and not self.carrying_survivor:
                nearest = min(active_survivors, key=lambda a: (a.x - self.x) ** 2 + (a.y - self.y) ** 2)
                current_distance = np.sqrt((nearest.x - self.x) ** 2 + (nearest.y - self.y) ** 2)

                if current_distance < 25:
                    if nearest.health_state == "healthy":
                        nearest.health_state = "injured"
                        reward += 45
                        nearest.cooldown = 15
                    elif nearest.health_state == "injured":
                        nearest.being_carried = True
                        self.carrying_survivor = nearest
                        reward += 70

        return reward

    async def prune_table(self, min_visits=3):
        return await self.q_table.prune(min_visits)


def draw_agent(surface, agent):
    if agent.on_hook:
        return

    if agent.caught:
        pygame.draw.circle(surface, SURVIVOR_CAUGHT_COLOR, (int(agent.x), int(agent.y)), agent.radius)
    elif agent.escaped:
        pygame.draw.circle(surface, SURVIVOR_ESCAPED_COLOR, (int(agent.x), int(agent.y)), agent.radius)
    elif agent.being_carried:
        pygame.draw.circle(surface, SURVIVOR_CARRIED_COLOR, (int(agent.x), int(agent.y)), agent.radius)
    elif agent.in_locker:
        pygame.draw.circle(surface, SURVIVOR_IN_LOCKER_COLOR, (int(agent.x), int(agent.y)), agent.radius)
    else:
        if agent.health_state == "injured":
            color = SURVIVOR_INJURED_COLOR
        elif agent.health_state == "dying":
            color = SURVIVOR_DYING_COLOR
        else:
            color = SURVIVOR_FIXING_COLOR if agent.fixing_generator else agent.color

        pygame.draw.circle(surface, color, (int(agent.x), int(agent.y)), agent.radius)


def create_agents(walls=[], loaded_survivors_data=None, loaded_hunter_data=None):
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
                q_table_data = None
                if loaded_survivors_data and i < len(loaded_survivors_data):
                    q_table_data = loaded_survivors_data[i]
                survivors.append(Agent(x, y, SURVIVOR_COLOR, False, i, q_table_data))
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
        hunter = Agent(hunter_x, hunter_y, HUNTER_COLOR, True, 0, loaded_hunter_data)
    return survivors, hunter


async def agent_step(agent, all_agents, generators, exits, hooks, lockers, totems, walls, generators_fixed, epsilon):
    state = await agent.get_state(all_agents, generators, exits, hooks, lockers, totems, generators_fixed)
    action = await agent.choose_action(state, epsilon)
    reward = await agent.move(action, all_agents, generators, exits, hooks, lockers, totems, walls)
    next_state = await agent.get_state(all_agents, generators, exits, hooks, lockers, totems, generators_fixed)
    await agent.update_q(state, action, reward, next_state)


async def simulate_agents_parallel(agents, generators, exits, hooks, lockers, totems, walls, generators_fixed, epsilon):
    tasks = []
    for agent in agents:
        if agent.caught or agent.escaped or agent.on_hook:
            continue

        task = asyncio.create_task(
            agent_step(agent, agents, generators, exits, hooks, lockers, totems, walls, generators_fixed, epsilon))
        tasks.append(task)

    await asyncio.gather(*tasks)