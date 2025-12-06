import pygame
import asyncio
import pickle
import os
import time
import logging
import numpy as np
import random
from config import *
from environment import EnvironmentGenerator
from agents import AsyncAgent
from renderer import Renderer

logger = logging.getLogger('DBD_QL_ASYNC')


class GameController:
    """Основной контроллер игры"""

    def __init__(self):
        self.screen = None
        self.clock = None
        self.font = None
        self.title_font = None

        self.walls = []
        self.exits = []
        self.survivors = []
        self.hunter = None
        self.generators = []

        self.episode = 0
        self.successful_escapes = 0

        self.running = True
        self.paused = False
        self.simulation_speed = DEFAULT_SIMULATION_SPEED
        self.show_vision_cones = True

    def initialize(self):
        """Инициализация игры"""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Dead by Daylight DQN - Deep Reinforcement Learning")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont('arial', 20)
        self.title_font = pygame.font.SysFont('arial', 24, bold=True)

        # Создание игрового мира через шум Перлина
        self.walls, self.generators = EnvironmentGenerator.create_perlin_map(5)
        self.exits = EnvironmentGenerator.create_random_exits(2, self.walls)
        self.survivors, self.hunter = self._create_agents(self.walls)

        logger.info("Игра инициализирована")

    def _create_agents(self, walls):
        """Создание агентов"""
        survivors = []
        positions = []
        num_survivors = NUM_SURVIVORS[0]  # Всегда 4 выживших

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
                    survivors.append(AsyncAgent(x, y, SURVIVOR_COLOR, False, i))
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
            hunter = AsyncAgent(hunter_x, hunter_y, HUNTER_COLOR, True)

        return survivors, hunter

    async def save_models_async(self):
        """Асинхронное сохранение моделей DQN"""
        try:
            # Сохраняем выживших
            loop = asyncio.get_event_loop()
            for i, agent in enumerate(self.survivors):
                filepath = f"survivor_{i}_dqn.pth"
                await loop.run_in_executor(None, agent.dqn_agent.save, filepath)

            # Сохраняем охотника
            if self.hunter is not None:
                await loop.run_in_executor(None, self.hunter.dqn_agent.save, "hunter_dqn.pth")

            # Сохраняем метаданные
            epsilon_avg = np.mean([a.dqn_agent.epsilon for a in self.survivors]) if self.survivors else 0.0
            metadata = {
                'episode': self.episode,
                'successful_escapes': self.successful_escapes,
                'epsilon': epsilon_avg,
                'timestamp': time.time(),
                'version': 3
            }
            await loop.run_in_executor(None, lambda: pickle.dump(metadata, open("dqn_metadata.pkl", 'wb'),
                                                                 protocol=pickle.HIGHEST_PROTOCOL))

            logger.info(
                f"Сохранено! Эпизод: {self.episode}, Побеги: {self.successful_escapes}")

        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")

    async def load_models_async(self):
        """Асинхронная загрузка моделей DQN"""
        loaded_episode = 0
        loaded_escapes = 0
        loaded_epsilon = 1.0

        # Загружаем метаданные
        if os.path.exists("dqn_metadata.pkl"):
            try:
                loop = asyncio.get_event_loop()
                metadata = await loop.run_in_executor(None, lambda: pickle.load(open("dqn_metadata.pkl", 'rb')))
                if metadata.get('version', 1) >= 3:
                    loaded_episode = metadata.get('episode', 0)
                    loaded_escapes = metadata.get('successful_escapes', 0)
                    loaded_epsilon = metadata.get('epsilon', 1.0)
            except Exception as e:
                logger.error(f"Ошибка загрузки метаданных: {e}")

        # Загружаем выживших
        loop = asyncio.get_event_loop()
        for i, agent in enumerate(self.survivors):
            filepath = f"survivor_{i}_dqn.pth"
            if os.path.exists(filepath):
                try:
                    await loop.run_in_executor(None, agent.dqn_agent.load, filepath)
                    logger.info(f"Загружен выживший {i}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки выжившего {i}: {e}")

        # Загружаем охотника
        if self.hunter is not None and os.path.exists("hunter_dqn.pth"):
            try:
                await loop.run_in_executor(None, self.hunter.dqn_agent.load, "hunter_dqn.pth")
                logger.info("Загружен охотник!")
            except Exception as e:
                logger.error(f"Ошибка загрузки охотника: {e}")

        return loaded_episode, loaded_escapes, loaded_epsilon

    async def reset_episode_async(self):
        """Сброс эпизода"""
        # Сначала подсчитываем, сколько выживших сбежало в этом эпизоде
        escaped_this_episode = sum(1 for s in self.survivors if s.escaped)
        if escaped_this_episode > 0:
            logger.info(f"В эпизоде {self.episode} сбежало: {escaped_this_episode} выживших")

        for survivor in self.survivors:
            attempts = 0
            while attempts < 50:
                x = random.randint(100, WIDTH - 100)
                y = random.randint(100, HEIGHT - 100)

                in_wall = False
                for wall in self.walls:
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
                    survivor.position_history = []
                    survivor.stuck_counter = 0
                    survivor.consecutive_wall_hits = 0
                    survivor.escape_mode = False
                    survivor.escape_steps = 0
                    survivor.movement_pattern = []
                    break
                attempts += 1

        if self.hunter is not None:
            hunter_x, hunter_y = WIDTH // 2, HEIGHT // 2
            for wall in self.walls:
                closest_x = max(wall.rect.left, min(hunter_x, wall.rect.right))
                closest_y = max(wall.rect.top, min(hunter_y, wall.rect.bottom))
                distance_sq = (hunter_x - closest_x) ** 2 + (hunter_y - closest_y) ** 2
                if distance_sq < 900:
                    hunter_x = random.randint(100, WIDTH - 100)
                    hunter_y = random.randint(100, HEIGHT - 100)
                    break

            self.hunter.x = hunter_x
            self.hunter.y = hunter_y
            self.hunter.caught = False
            self.hunter.escaped = False
            self.hunter.cooldown = 0
            self.hunter.capture_target = None
            self.hunter.hold_steps = 0
            self.hunter.last_distance_to_target = float('inf')
            self.hunter.position_history = []
            self.hunter.stuck_counter = 0
            self.hunter.consecutive_wall_hits = 0
            self.hunter.escape_mode = False
            self.hunter.escape_steps = 0
            self.hunter.movement_pattern = []

        # Генераторы уже созданы в create_perlin_map

    async def update_agents_async(self, generators_fixed):
        """Асинхронное обновление агентов"""
        all_agents = self.survivors
        if self.hunter is not None:
            all_agents = self.survivors + [self.hunter]

        tasks = []

        for agent in all_agents:
            if agent.caught or agent.escaped:
                continue

            task = asyncio.create_task(self._update_single_agent_async(
                agent, all_agents, generators_fixed
            ))
            tasks.append(task)

            if len(tasks) >= ASYNC_BATCH_SIZE:
                await asyncio.gather(*tasks)
                await asyncio.sleep(ASYNC_SLEEP_TIME)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)

    async def _update_single_agent_async(self, agent, all_agents, generators_fixed):
        """Обновление одного агента с DQN"""
        if agent.caught or agent.escaped:
            return

        state = await agent.get_state_vector(all_agents, self.generators, self.exits, generators_fixed, self.walls)
        action = agent.dqn_agent.act(state, training=True)
        agent.last_action = action
        
        reward, done = await agent.smooth_move_async(action, all_agents, self.generators, self.walls)
        
        next_state = await agent.get_state_vector(all_agents, self.generators, self.exits, generators_fixed, self.walls)
        
        # Сохраняем опыт в replay buffer
        agent.dqn_agent.remember(state, action, reward, next_state, done)
        
        # Обучение на батче (не каждый шаг для оптимизации)
        if len(agent.dqn_agent.memory) > agent.dqn_agent.batch_size and agent.dqn_agent.training_steps % 4 == 0:
            agent.dqn_agent.replay()

        # Проверка побега - только если агент еще не сбежал
        if not agent.is_hunter and not agent.escaped and generators_fixed == len(self.generators):
            for exit_pos in self.exits:
                if (exit_pos[0] - agent.x) ** 2 + (exit_pos[1] - agent.y) ** 2 < 1225:  # 35^2 = 1225
                    agent.escaped = True
                    self.successful_escapes += 1
                    logger.info(f"Выживший {agent.agent_id} сбежал! Всего побегов: {self.successful_escapes}")
                    break  # Выходим из цикла после первого подходящего выхода

    async def handle_events(self):
        """Асинхронная обработка событий"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                await self._handle_keydown(event)

    async def _handle_keydown(self, event):
        """Обработка нажатий клавиш"""
        if event.key == pygame.K_SPACE:
            self.paused = not self.paused
            logger.info("Пауза" if self.paused else "Продолжение")
        elif event.key == pygame.K_s:
            await self.save_models_async()
        elif event.key == pygame.K_l:
            self.episode, self.successful_escapes, _ = await self.load_models_async()
        elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
            self.simulation_speed = min(self.simulation_speed * 2, 10000)
            logger.info(f"Скорость симуляции: {self.simulation_speed:.0f}x")
        elif event.key == pygame.K_MINUS:
            self.simulation_speed = max(self.simulation_speed / 2, 0.125)
            logger.info(f"Скорость симуляции: {self.simulation_speed:.0f}x")
        elif event.key == pygame.K_0:
            self.simulation_speed = 1.0
            logger.info("Нормальная скорость")
        elif event.key == pygame.K_v:
            self.show_vision_cones = not self.show_vision_cones
            logger.info(f"Конусы зрения: {'вкл' if self.show_vision_cones else 'выкл'}")

    def render(self):
        """Отрисовка игры"""
        self.screen.fill(BACKGROUND)

        # Отрисовка стен
        for wall in self.walls:
            wall.draw(self.screen)

        # Отрисовка выходов
        exit_active = self.get_generators_fixed() == len(self.generators)
        for exit_pos in self.exits:
            color = (0, 255, 0) if exit_active else EXIT_COLOR
            size = 18 if exit_active else 15
            pygame.draw.circle(self.screen, color, exit_pos, size)
            if exit_active and pygame.time.get_ticks() % 1000 < 500:
                pygame.draw.circle(self.screen, (255, 255, 255), exit_pos, size, 2)

        # Отрисовка генераторов
        for generator in self.generators:
            generator.draw(self.screen)

        # Отрисовка конусов зрения
        if self.show_vision_cones:
            for agent in self.survivors:
                if not agent.caught and not agent.escaped:
                    Renderer.draw_vision_cone(self.screen, agent)
            if self.hunter is not None and not self.hunter.caught and not self.hunter.escaped:
                Renderer.draw_vision_cone(self.screen, self.hunter)

        # Отрисовка агентов
        for agent in self.survivors:
            Renderer.draw_agent(self.screen, agent)
        if self.hunter is not None:
            Renderer.draw_agent(self.screen, self.hunter)

        # Отрисовка UI
        self._draw_ui()

        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_ui(self):
        """Отрисовка пользовательского интерфейса"""
        # Левая панель
        ui_left = Renderer.create_transparent_surface(400, 280, UI_BG)
        self.screen.blit(ui_left, (10, 10))

        active_survivors = sum(1 for s in self.survivors if not s.caught and not s.escaped)
        stuck_survivors = sum(1 for s in self.survivors if s.escape_mode)
        escaped_survivors = sum(1 for s in self.survivors if s.escaped)
        caught_survivors = sum(1 for s in self.survivors if s.caught)

        if self.simulation_speed >= 1:
            speed_display = f"{self.simulation_speed:.0f}x"
        else:
            speed_display = f"1/{1 / self.simulation_speed:.0f}x"

        avg_reward = np.mean(self.survivors[0].last_rewards) if self.survivors and self.survivors[0].last_rewards else 0
        hunter_avg = np.mean(self.hunter.last_rewards) if self.hunter is not None and self.hunter.last_rewards else 0

        memory_size = len(self.survivors[0].dqn_agent.memory) if self.survivors else 0
        epsilon_dqn = self.survivors[0].dqn_agent.epsilon if self.survivors else 0
        
        stats = [
            f"DQN - Deep Reinforcement Learning",
            f"Эпизод: {self.episode}/{EPISODES}",
            f"Всего побегов: {self.successful_escapes}",
            f"Генераторы: {self.get_generators_fixed()}/{len(self.generators)}",
            f"Активные: {active_survivors}/{len(self.survivors)}",
            f"Сбежали: {escaped_survivors}",
            f"Пойманы: {caught_survivors}",
            f"Застрявшие: {stuck_survivors}",
            f"Охотник: {'вкл' if self.hunter is not None else 'выкл'}",
            f"ε: {epsilon_dqn:.3f}",
            f"Скорость: {speed_display}",
            f"Награда выжившего: {avg_reward:.2f}",
            f"Награда охотника: {hunter_avg:.2f}",
            f"Replay Buffer: {memory_size}"
        ]

        for i, text in enumerate(stats):
            text_color = TEXT_COLOR
            if i == 2:  # Подсвечиваем строку с общим количеством побегов
                text_color = (0, 255, 0) if self.successful_escapes > 0 else TEXT_COLOR
            elif i == 5:  # Подсвечиваем строку с сбежавшими в текущем эпизоде
                text_color = (0, 200, 0) if escaped_survivors > 0 else TEXT_COLOR
            elif i == 6:  # Подсвечиваем строку с пойманными
                text_color = (255, 100, 100) if caught_survivors > 0 else TEXT_COLOR
            elif i == 7:  # Подсвечиваем строку с застрявшими
                text_color = (255, 100, 100) if stuck_survivors > 0 else TEXT_COLOR

            text_surface = self.font.render(text, True, text_color)
            self.screen.blit(text_surface, (20, 15 + i * 25))

        # Правая панель (управление)
        ui_right = Renderer.create_transparent_surface(250, 200, UI_BG)
        self.screen.blit(ui_right, (WIDTH - 260, 10))

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
            text_surface = self.font.render(text, True, TEXT_COLOR)
            self.screen.blit(text_surface, (WIDTH - 250, 20 + i * 25))

        # Пауза
        if self.paused:
            pause_ui = Renderer.create_transparent_surface(500, 60, UI_BG)
            self.screen.blit(pause_ui, (WIDTH // 2 - 250, HEIGHT // 2 - 30))
            pause_text = self.title_font.render("ПАУЗА - Нажмите ПРОБЕЛ для продолжения", True, (255, 50, 50))
            self.screen.blit(pause_text, (WIDTH // 2 - pause_text.get_width() // 2, HEIGHT // 2 - 10))

        # Скорость
        if self.simulation_speed != 1.0:
            speed_ui = Renderer.create_transparent_surface(200, 40, UI_BG)
            self.screen.blit(speed_ui, (WIDTH // 2 - 100, 10))
            speed_text = self.font.render(f"СКОРОСТЬ: {speed_display}", True, (255, 165, 0))
            self.screen.blit(speed_text, (WIDTH // 2 - speed_text.get_width() // 2, 20))

    def get_generators_fixed(self):
        """Количество починенных генераторов"""
        return sum(1 for g in self.generators if g.fixed)

    async def run(self):
        """Главный игровой цикл"""
        self.initialize()
        self.episode, self.successful_escapes, _ = await self.load_models_async()

        logger.info(f"Начало обучения с эпизода {self.episode}, предыдущих побед: {self.successful_escapes}")

        while self.running and self.episode < EPISODES:
            await self.handle_events()  # Теперь асинхронный вызов

            if not self.paused:
                generators_fixed = self.get_generators_fixed()

                # Асинхронное обновление
                steps = max(1, int(self.simulation_speed))
                for step in range(steps):
                    await self.update_agents_async(generators_fixed)

                    # Проверка завершения эпизода
                    if all(s.escaped or s.caught for s in self.survivors):
                        self.episode += 1
                        # Epsilon управляется внутри DQN агентов

                        # Периодическое сохранение
                        if self.episode % 50 == 0:
                            await self.save_models_async()

                            # Статистика
                            avg_reward = np.mean(self.survivors[0].last_rewards) if self.survivors and self.survivors[
                                0].last_rewards else 0
                            hunter_avg = np.mean(
                                self.hunter.last_rewards) if self.hunter is not None and self.hunter.last_rewards else 0
                            stuck_count = sum(1 for a in self.survivors if a.escape_mode)
                            escaped_count = sum(1 for a in self.survivors if a.escaped)
                            caught_count = sum(1 for a in self.survivors if a.caught)
                            memory_sizes = [len(a.dqn_agent.memory) for a in self.survivors]
                            if self.hunter is not None:
                                memory_sizes.append(len(self.hunter.dqn_agent.memory))

                            logger.info(
                                f"Эпизод {self.episode}, ε={self.survivors[0].dqn_agent.epsilon:.3f}, Всего побед: {self.successful_escapes}, "
                                f"Сбежали: {escaped_count}, Пойманы: {caught_count}, "
                                f"Застрявших: {stuck_count}, Replay Buffer размеры: {memory_sizes}")

                        await self.reset_episode_async()
                        break

            self.render()

        logger.info("Завершение обучения...")
        await self.save_models_async()
        pygame.quit()
