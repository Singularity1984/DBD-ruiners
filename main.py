from Libary import *
from Agent import *
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DBD_QL')

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dead by Daylight Q-Learning - ASYNC")
clock = pygame.time.Clock()


# ДОБАВЛЕНА ФУНКЦИЯ ДЛЯ ОТРИСОВКИ ПОДПИСЕЙ
def draw_label(surface, text, x, y, font_size=LABEL_FONT_SIZE):
    """Рисует подпись с черным фоном над объектом"""
    font = pygame.font.SysFont('Arial', font_size)
    text_surface = font.render(text, True, LABEL_COLOR)
    text_rect = text_surface.get_rect(center=(x, y))

    # Рисуем фон под текстом
    bg_rect = text_rect.inflate(10, 6)
    bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
    bg_surface.fill(LABEL_BACKGROUND)
    surface.blit(bg_surface, bg_rect)

    # Рисуем текст
    surface.blit(text_surface, text_rect)


class Wall:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, WALL_COLOR, self.rect)
        pygame.draw.rect(surface, WALL_BORDER_COLOR, self.rect, 2)


class Generator:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.progress = 0
        self.fixed = False
        self.radius = 25
        self.repair_radius = 35
        self._lock = asyncio.Lock()

    async def repair(self, amount):
        async with self._lock:
            if not self.fixed:
                self.progress += amount
                if self.progress >= 100:
                    self.fixed = True
                    return True
            return False

    def draw(self, surface, show_labels=False):
        color = GENERATOR_FIXED_COLOR if self.fixed else GENERATOR_COLOR
        pygame.draw.circle(surface, color, (self.x, self.y), self.radius)

        if not self.fixed:
            pygame.draw.circle(surface, (100, 100, 100), (self.x, self.y), self.radius, 2)
            progress_angle = int(360 * (self.progress / 100))
            if progress_angle > 0:
                pygame.draw.arc(surface, GENERATOR_PROGRESS_COLOR,
                                (self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2),
                                0, np.radians(progress_angle), 4)

        # ДОБАВЛЕНА ПОДПИСЬ
        if show_labels:
            label = "Генератор (починен)" if self.fixed else f"Генератор ({self.progress}%)"
            draw_label(surface, label, self.x, self.y - self.radius - 15)


class Hook:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.radius = 25
        self.survivor = None
        self.progress = 0
        self.stage = 1
        self.available = True
        self._lock = asyncio.Lock()

    async def add_survivor(self, survivor):
        async with self._lock:
            if self.available:
                self.survivor = survivor
                self.available = False
                return True
            return False

    async def remove_survivor(self):
        async with self._lock:
            self.survivor = None
            self.progress = 0
            self.available = True

    async def update_progress(self, amount):
        async with self._lock:
            if self.survivor:
                self.progress += amount
                # ИСПРАВЛЕНО: используем настройку из конфига
                if self.progress >= HOOK_STAGE_DURATION:
                    if self.stage == 1:
                        self.stage = 2
                        self.progress = 0
                        return "stage_advance"
                    else:
                        result = self.survivor
                        self.survivor = None
                        self.progress = 0
                        self.stage = 1
                        self.available = True
                        return "sacrificed"
            return "continue"

    def draw(self, surface, show_labels=False):
        color = HOOK_COLOR if self.available else HOOK_OCCUPIED_COLOR
        pygame.draw.circle(surface, color, (self.x, self.y), self.radius)

        if self.survivor:
            progress_color = HOOK_PROGRESS_COLOR if self.progress > 50 else HOOK_PROGRESS_WARNING_COLOR
            # ИСПРАВЛЕНО: используем настройку из конфига для расчета угла
            progress_angle = int(360 * (self.progress / HOOK_STAGE_DURATION))
            pygame.draw.arc(surface, progress_color,
                            (self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2),
                            0, np.radians(progress_angle), 4)

        # ДОБАВЛЕНА ПОДПИСЬ
        if show_labels:
            if self.survivor:
                label = f"Крюк (жертва: {int(self.progress)}%)"
            else:
                label = "Крюк (свободен)"
            draw_label(surface, label, self.x, self.y - self.radius - 15)

class Locker:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id
        self.width = 40
        self.height = 60
        self.occupied = False
        self.rect = pygame.Rect(x - self.width // 2, y - self.height // 2, self.width, self.height)
        self._lock = asyncio.Lock()

    async def enter(self):
        async with self._lock:
            if not self.occupied:
                self.occupied = True
                return True
            return False

    async def exit(self):
        async with self._lock:
            self.occupied = False

    def draw(self, surface, show_labels=False):
        color = LOCKER_COLOR if not self.occupied else LOCKER_OCCUPIED_COLOR
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, LOCKER_BORDER_COLOR, self.rect, 2)

        # ДОБАВЛЕНА ПОДПИСЬ
        if show_labels:
            label = "Шкаф (занят)" if self.occupied else "Шкаф (свободен)"
            draw_label(surface, label, self.x, self.y - self.height // 2 - 15)


class Totem:
    def __init__(self, x, y, id, is_hex=False):
        self.x = x
        self.y = y
        self.id = id
        self.active = True
        self.is_hex = is_hex
        self.radius = 15
        self._lock = asyncio.Lock()

    async def cleanse(self):
        async with self._lock:
            if self.active:
                self.active = False
                return True
            return False

    def draw(self, surface, show_labels=False):
        if self.is_hex:
            color = TOTEM_HEX_COLOR if self.active else TOTEM_CLEANSED_COLOR
        else:
            color = TOTEM_COLOR if self.active else TOTEM_CLEANSED_COLOR
        pygame.draw.circle(surface, color, (self.x, self.y), self.radius)

        # ДОБАВЛЕНА ПОДПИСЬ С ОБЪЯСНЕНИЕМ
        if show_labels:
            if self.is_hex:
                label = "Проклятый тотем" if self.active else "Тотем (очищен)"
            else:
                label = "Обычный тотем" if self.active else "Тотем (очищен)"
            draw_label(surface, label, self.x, self.y - self.radius - 15)


def create_transparent_surface(width, height, color):
    surface = pygame.Surface((width, height), pygame.SRCALPHA)
    surface.fill(color)
    return surface


async def save_models(survivors, hunter, episode, successful_escapes, epsilon):
    try:
        survivors_data = []
        for agent in survivors:
            q_data = {}
            for key in list(agent.q_table._data.keys()):
                q_data[key] = (await agent.q_table.get(key)).copy()
            survivors_data.append(q_data)

        data_survivors = {
            'survivors_q_tables': survivors_data,
            'episode': episode,
            'successful_escapes': successful_escapes,
            'epsilon': epsilon,
            'timestamp': time.time(),
            'version': 4
        }

        async with aiofiles.open(SURVIVORS_SAVE_FILE, 'wb') as f:
            await f.write(pickle.dumps(data_survivors))

        if hunter is not None:
            hunter_data = {}
            for key in list(hunter.q_table._data.keys()):
                hunter_data[key] = (await hunter.q_table.get(key)).copy()

            data_hunter = {
                'hunter_q_table': hunter_data,
                'episode': episode,
                'timestamp': time.time(),
                'version': 4
            }

            async with aiofiles.open(HUNTER_SAVE_FILE, 'wb') as f:
                await f.write(pickle.dumps(data_hunter))

        logger.info(f"Сохранено! Эпизод: {episode}")

    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")


async def load_models():
    loaded_episode = 0
    loaded_escapes = 0
    loaded_epsilon = 1.0
    loaded_survivors_data = None
    loaded_hunter_data = None

    if os.path.exists(SURVIVORS_SAVE_FILE):
        try:
            async with aiofiles.open(SURVIVORS_SAVE_FILE, 'rb') as f:
                data = pickle.loads(await f.read())

            if data.get('version', 1) >= 2:
                loaded_survivors_data = data['survivors_q_tables']
                loaded_episode = data['episode']
                loaded_escapes = data['successful_escapes']
                loaded_epsilon = data['epsilon']
                logger.info(f"Загружены выжившие! Эпизод: {loaded_episode}, ε: {loaded_epsilon:.3f}")

        except Exception as e:
            logger.error(f"Ошибка загрузки выживших: {e}")

    if os.path.exists(HUNTER_SAVE_FILE):
        try:
            async with aiofiles.open(HUNTER_SAVE_FILE, 'rb') as f:
                data = pickle.loads(await f.read())

            if data.get('version', 1) >= 2:
                loaded_hunter_data = data['hunter_q_table']
                logger.info("Загружен охотник!")

        except Exception as e:
            logger.error(f"Ошибка загрузки охотника: {e}")

    return loaded_episode, loaded_escapes, loaded_epsilon, loaded_survivors_data, loaded_hunter_data


async def update_hooks_parallel(hooks):
    tasks = []
    for hook in hooks:
        # ИСПРАВЛЕНО: используем настройку из конфига
        task = asyncio.create_task(hook.update_progress(HOOK_PROGRESS_SPEED))
        tasks.append(task)
    await asyncio.gather(*tasks)


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


def create_hooks(walls=[]):
    """Создает крюки в случайных местах"""
    count = random.randint(NUM_HOOKS[0], NUM_HOOKS[1])  # ИСПРАВЛЕНО: используем настройки из конфига
    hooks = []
    for i in range(count):
        placed = False
        attempts = 0
        while not placed and attempts < 50:
            x = random.randint(100, WIDTH - 100)
            y = random.randint(100, HEIGHT - 100)
            in_wall = False
            for wall in walls:
                if wall.rect.collidepoint(x, y):
                    in_wall = True
                    break
            if not in_wall:
                hooks.append(Hook(x, y, i))
                placed = True
            attempts += 1
    return hooks


def create_lockers(walls=[]):
    """Создает шкафы в случайных местах"""
    count = random.randint(NUM_LOCKERS[0], NUM_LOCKERS[1])  # ИСПРАВЛЕНО: используем настройки из конфига
    lockers = []
    for i in range(count):
        placed = False
        attempts = 0
        while not placed and attempts < 50:
            x = random.randint(100, WIDTH - 100)
            y = random.randint(100, HEIGHT - 100)
            in_wall = False
            for wall in walls:
                if wall.rect.collidepoint(x, y):
                    in_wall = True
                    break
            if not in_wall:
                lockers.append(Locker(x, y, i))
                placed = True
            attempts += 1
    return lockers


def create_totems(walls=[]):
    """Создает тотемы в случайных местах"""
    count = random.randint(NUM_TOTEMS[0], NUM_TOTEMS[1])  # ИСПРАВЛЕНО: используем настройки из конфига
    hex_count = random.randint(NUM_HEX_TOTEMS[0], NUM_HEX_TOTEMS[1])  # ИСПРАВЛЕНО: используем настройки из конфига
    totems = []
    for i in range(count):
        placed = False
        attempts = 0
        while not placed and attempts < 50:
            x = random.randint(100, WIDTH - 100)
            y = random.randint(100, HEIGHT - 100)
            in_wall = False
            for wall in walls:
                if wall.rect.collidepoint(x, y):
                    in_wall = True
                    break
            if not in_wall:
                is_hex = (i < hex_count)
                totems.append(Totem(x, y, i, is_hex))
                placed = True
            attempts += 1
    return totems


async def main_game_loop():
    walls = create_random_walls()
    exits = create_random_exits(None, walls)
    hooks = create_hooks(walls)  # ИСПРАВЛЕНО: убрал жесткое число
    lockers = create_lockers(walls)  # ИСПРАВЛЕНО: убрал жесткое число
    totems = create_totems(walls)  # ИСПРАВЛЕНО: убрал жесткое число

    # ИСПРАВЛЕННАЯ ЗАГРУЗКА: сначала загружаем данные, потом создаем агентов
    loaded_episode, loaded_escapes, loaded_epsilon, loaded_survivors_data, loaded_hunter_data = await load_models()

    survivors, hunter = create_agents(walls, loaded_survivors_data, loaded_hunter_data)
    generators = create_random_generators(None, 150, walls)

    font = pygame.font.SysFont('arial', 20)
    title_font = pygame.font.SysFont('arial', 24, bold=True)

    episode, successful_escapes, epsilon = loaded_episode, loaded_escapes, loaded_epsilon
    last_save_episode = episode

    running = True
    paused = False
    simulation_speed = DEFAULT_SIMULATION_SPEED
    show_vision_cones = True
    show_labels = True  # НОВАЯ ПЕРЕМЕННАЯ ДЛЯ УПРАВЛЕНИЯ ПОДПИСЯМИ

    last_log_time = time.time()
    frame_count = 0

    logger.info(f"Начало обучения с эпизода {episode}")

    while running and episode < EPISODES:
        current_time = time.time()
        frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_s:
                    await save_models(survivors, hunter, episode, successful_escapes, epsilon)
                elif event.key == pygame.K_l:
                    # Перезагружаем модели
                    loaded_episode, loaded_escapes, loaded_epsilon, loaded_survivors_data, loaded_hunter_data = await load_models()
                    survivors, hunter = create_agents(walls, loaded_survivors_data, loaded_hunter_data)
                    episode, successful_escapes, epsilon = loaded_episode, loaded_escapes, loaded_epsilon
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    simulation_speed = min(simulation_speed * 2, 10000)
                elif event.key == pygame.K_MINUS:
                    simulation_speed = max(simulation_speed / 2, 0.125)
                elif event.key == pygame.K_0:
                    simulation_speed = 1.0
                elif event.key == pygame.K_v:
                    show_vision_cones = not show_vision_cones
                elif event.key == pygame.K_n:  # НОВАЯ КЛАВИША ДЛЯ ПЕРЕКЛЮЧЕНИЯ ПОДПИСЕЙ
                    show_labels = not show_labels

        if not paused:
            generators_fixed = sum(1 for g in generators if g.fixed)
            steps = min(max(1, int(simulation_speed)), MAX_STEPS_PER_FRAME)

            # ДОБАВЛЕНО: Подсчет активных тотемов и применение бонусов охотнику
            active_hex_totems = sum(1 for totem in totems if totem.active and totem.is_hex)
            active_normal_totems = sum(1 for totem in totems if totem.active and not totem.is_hex)

            if hunter:
                # Сбрасываем бонусы к базовым значениям
                hunter.speed = HUNTER_SPEED
                hunter.vision_radius = HUNTER_VISION_RADIUS

                # Применяем бонусы от тотемов
                if active_hex_totems > 0:
                    hunter.speed *= (1 + HEX_TOTEM_SPEED_BONUS)
                    hunter.vision_radius += HEX_TOTEM_VISION_BONUS

                if active_normal_totems > 0:
                    hunter.speed *= (1 + NORMAL_TOTEM_SPEED_BONUS)

            for _ in range(steps):
                all_agents = survivors + ([hunter] if hunter else [])

                await simulate_agents_parallel(all_agents, generators, exits, hooks, lockers, totems, walls,
                                               generators_fixed, epsilon)
                await update_hooks_parallel(hooks)

                escaped_count = sum(1 for s in survivors if s.escaped)
                caught_count = sum(1 for s in survivors if s.caught)

                # УЛУЧШЕНИЕ: Награда за успешный побег
                if escaped_count > 0:
                    successful_escapes += 1
                    logger.info(f"УСПЕШНЫЙ ПОБЕГ! Всего побед: {successful_escapes}")

                if escaped_count + caught_count >= len(survivors):
                    episode += 1

                    # УЛУЧШЕНИЕ: Более медленное уменьшение epsilon в начале
                    if episode < 1000:
                        epsilon = max(0.3, epsilon * EPSILON_DECAY)  # Минимальный epsilon 0.3 для исследования
                    else:
                        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

                    if episode % PRUNE_EVERY == 0 and episode > 0:
                        prune_tasks = []
                        for agent in all_agents:
                            prune_tasks.append(asyncio.create_task(agent.prune_table()))
                        await asyncio.gather(*prune_tasks)

                    if episode % 50 == 0:
                        await save_models(survivors, hunter, episode, successful_escapes, epsilon)
                        last_save_episode = episode

                        if current_time - last_log_time >= 1.0:
                            logger.info(
                                f"Эпизод {episode}, ε={epsilon:.3f}, Побед: {successful_escapes}, Скорость: {simulation_speed:.0f}x")
                            last_log_time = current_time

                    # Пересоздаем агентов и генераторы
                    survivors, hunter = create_agents(walls, loaded_survivors_data, loaded_hunter_data)
                    generators = create_random_generators(None, 150, walls)
                    break

        if simulation_speed <= DISABLE_VISUALIZATION_ABOVE:
            screen.fill(BACKGROUND)

            for wall in walls:
                wall.draw(screen)

            # ОБНОВЛЕНА ОТРИСОВКА С ПОДПИСЯМИ
            should_show_labels = show_labels and simulation_speed <= SHOW_LABELS_BELOW_SPEED

            for hook in hooks:
                hook.draw(screen, should_show_labels)
            for locker in lockers:
                locker.draw(screen, should_show_labels)
            for totem in totems:
                totem.draw(screen, should_show_labels)

            exit_active = generators_fixed == len(generators)
            for exit_pos in exits:
                color = EXIT_ACTIVE_COLOR if exit_active else EXIT_COLOR
                size = 18 if exit_active else 15
                pygame.draw.circle(screen, color, exit_pos, size)

                # ПОДПИСЬ ДЛЯ ВЫХОДА
                if should_show_labels:
                    label = "Выход (активен)" if exit_active else "Выход (неактивен)"
                    draw_label(screen, label, exit_pos[0], exit_pos[1] - 25)

            for generator in generators:
                generator.draw(screen, should_show_labels)

            for agent in survivors:
                draw_agent(screen, agent)
            if hunter:
                draw_agent(screen, hunter)

            if simulation_speed <= 200:
                ui_left = create_transparent_surface(400, 300, UI_BG)
                screen.blit(ui_left, (10, 10))

                active_survivors = sum(1 for s in survivors if not s.caught and not s.escaped and not s.on_hook)
                injured_survivors = sum(1 for s in survivors if s.health_state == "injured")
                hooked_survivors = sum(1 for s in survivors if s.on_hook)

                speed_display = f"{simulation_speed:.0f}x" if simulation_speed >= 1 else f"1/{1 / simulation_speed:.0f}x"

                stats = [
                    f"DBD ASYNC",
                    f"Эпизод: {episode}/{EPISODES}",
                    f"Побеги: {successful_escapes}",
                    f"Генераторы: {generators_fixed}/{len(generators)}",
                    f"Активные: {active_survivors}/{len(survivors)}",
                    f"Раненые: {injured_survivors}",
                    f"На крюках: {hooked_survivors}",
                    f"ε: {epsilon:.3f}",
                    f"Скорость: {speed_display}",
                    f"Подписи: {'ВКЛ' if show_labels else 'ВЫКЛ'} (N)",  # ДОБАВЛЕНА ИНФОРМАЦИЯ О ПОДПИСЯХ
                    f"Проклятые тотемы: {active_hex_totems}",  # ДОБАВЛЕНО
                    f"Обычные тотемы: {active_normal_totems}",  # ДОБАВЛЕНО
                ]

                for i, text in enumerate(stats):
                    text_color = TEXT_COLOR
                    text_surface = font.render(text, True, text_color)
                    screen.blit(text_surface, (20, 15 + i * 25))

                # ДОБАВЛЕНО: Показ бонусов охотника
                if hunter and simulation_speed <= 200:
                    bonus_text = []
                    if active_hex_totems > 0:
                        bonus_text.append(
                            f"Охотник: +{HEX_TOTEM_SPEED_BONUS * 100}% скорости, +{HEX_TOTEM_VISION_BONUS} зрения")
                    if active_normal_totems > 0:
                        bonus_text.append(f"Охотник: +{NORMAL_TOTEM_SPEED_BONUS * 100}% скорости")

                    for i, text in enumerate(bonus_text):
                        text_surface = font.render(text, True, (255, 100, 100))
                        screen.blit(text_surface, (20, 300 + i * 25))

            if paused and simulation_speed <= 100:
                pause_ui = create_transparent_surface(500, 60, UI_BG)
                screen.blit(pause_ui, (WIDTH // 2 - 250, HEIGHT // 2 - 30))
                pause_text = title_font.render("ПАУЗА - Нажмите ПРОБЕЛ для продолжения", True, (255, 50, 50))
                screen.blit(pause_text, (WIDTH // 2 - pause_text.get_width() // 2, HEIGHT // 2 - 10))

            pygame.display.flip()
        else:
            if frame_count % 10 == 0:
                screen.fill(BACKGROUND)
                info_text = [
                    f"ТУРБО: {simulation_speed:.0f}x",
                    f"Эпизод: {episode}",
                    f"Побеги: {successful_escapes}",
                ]
                for i, text in enumerate(info_text):
                    text_surface = font.render(text, True, TEXT_COLOR)
                    screen.blit(text_surface, (WIDTH // 2 - 100, HEIGHT // 2 - 50 + i * 30))
                pygame.display.flip()

        if simulation_speed > 500:
            clock.tick(30)
        elif simulation_speed > 200:
            clock.tick(45)
        elif simulation_speed > 100:
            clock.tick(60)
        else:
            clock.tick(FPS)

    await save_models(survivors, hunter, episode, successful_escapes, epsilon)
    pygame.quit()


if __name__ == "__main__":
    asyncio.run(main_game_loop())