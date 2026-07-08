# Game settings
WIDTH, HEIGHT = 1280, 720
BACKGROUND = (10, 15, 20)
SURVIVOR_COLOR = (30, 144, 255)
HUNTER_COLOR = (220, 20, 60)
GENERATOR_COLOR = (34, 139, 34)
EXIT_COLOR = (255, 215, 0)
WALL_COLOR = (80, 80, 100)
TEXT_COLOR = (240, 240, 240)
UI_BG = (50, 50, 50, 128)

# Q-Learning parameters
LEARNING_RATE = 0.2
DISCOUNT = 0.95
EPISODES = 5000
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.001

# Adaptive epsilon based on performance
USE_ADAPTIVE_EPSILON = True
EPSILON_SUCCESS_THRESHOLD = 0.3  # Если процент побегов > 30%, уменьшаем epsilon быстрее
EPSILON_SUCCESS_DECAY = 0.998  # Дополнительный decay при хорошей производительности

# Heuristic usage - уменьшаем влияние эвристики со временем
HEURISTIC_MAX_EPISODES = 200  # После этого эпизода эвристика используется реже
HEURISTIC_INITIAL_PROB = 0.8  # Вероятность использования эвристики в начале
HEURISTIC_FINAL_PROB = 0.1  # Минимальная вероятность использования эвристики

# Управление загрузкой на старте (чтобы начинать с 1 эпизода)
LOAD_ON_START = False

# Experience Replay
USE_EXPERIENCE_REPLAY = True
REPLAY_BUFFER_SIZE = 1000  # Размер буфера опыта
REPLAY_BATCH_SIZE = 32  # Размер батча для переобучения
REPLAY_UPDATE_FREQUENCY = 10  # Частота обновления из буфера (каждые N эпизодов)

# Capture parameters
CAPTURE_RADIUS = 40
CAPTURE_HOLD_STEPS = 12

# Memory limits
MAX_Q_TABLE_SIZE = 50000
PRUNE_EVERY = 500

# Сохранение в разные файлы
SURVIVORS_SAVE_FILE = "dbd_survivors.pkl"
HUNTER_SAVE_FILE = "dbd_hunter.pkl"

# Система уровней навыков
SKILL_LEVELS = {
    "novice": {"min_episodes": 0, "max_episodes": 500, "name": "Новичок"},
    "intermediate": {"min_episodes": 500, "max_episodes": 2000, "name": "Средний"},
    "advanced": {"min_episodes": 2000, "max_episodes": 5000, "name": "Продвинутый"},
    "master": {"min_episodes": 5000, "max_episodes": float('inf'), "name": "Мастер"}
}

# Метрики для оценки навыка
SKILL_EVALUATION_WINDOW = 100  # Окно эпизодов для оценки навыка

# Stuck avoidance
STUCK_THRESHOLD = 20
ESCAPE_BOOST = 0.3
POSITION_MEMORY = 50

# Agent parameters
SURVIVOR_SPEED = 3.0
HUNTER_SPEED = 2.8
SURVIVOR_RADIUS = 11
HUNTER_RADIUS = 14
SURVIVOR_VISION_RADIUS = 260
HUNTER_VISION_RADIUS = 260  # маньяк видит так же как выжившие
SURVIVOR_VISION_ANGLE = 150
HUNTER_VISION_ANGLE = 150  # маньяк видит так же как выжившие

# Generation parameters
NUM_SURVIVORS = (2, 2)  # фиксированно два выживших для тренировки
NUM_HUNTERS = 1
NUM_GENERATORS = (3, 3)  # 3 генератора в тренировочной комнате
NUM_EXITS = (2, 2)
NUM_WALLS = (12, 16)

# Простая тренировочная комната
USE_TRAINING_ROOM = False  # Используем генерацию карт через Perlin noise
TRAINING_ROOM_LAYOUT = {
    "survivors": [(220, HEIGHT // 2 - 40), (220, HEIGHT // 2 + 40)],
    "hunter": (WIDTH - 220, HEIGHT // 2),
    "generator": (WIDTH // 2, HEIGHT // 2),  # основной генератор в центре
    "generators": [
        (WIDTH // 2, HEIGHT // 2 - 220),  # генератор сверху (выше)
        (WIDTH // 2, HEIGHT // 2),        # генератор в центре (основной)
        (WIDTH // 2, HEIGHT // 2 + 220),  # генератор снизу (ниже)
    ],
    "exits": [(80, HEIGHT // 2), (WIDTH - 80, HEIGHT // 2)],
    # невысокие стены создают коридор и не блокируют видимость генератора
    "walls": [
        (WIDTH // 2 - 160, HEIGHT // 2 - 140, 40, 280),
        (WIDTH // 2 + 120, HEIGHT // 2 - 140, 40, 280),
        (WIDTH // 2 - 260, HEIGHT // 2 - 60, 80, 40),
        (WIDTH // 2 + 180, HEIGHT // 2 - 60, 80, 40),
    ],
}

# Simulation settings
ENABLE_HUNTER = True  # Охотник включен для комнаты-примера
DEFAULT_SIMULATION_SPEED = 1.0
FPS = 15

# Async settings
ASYNC_BATCH_SIZE = 5  # Количество агентов для параллельного обновления
ASYNC_SLEEP_TIME = 0.001  # Время сна между батчами для предотвращения блокировки

# Cookie rewards
COOKIE_SURVIVOR_SPOT = 1        # за то, что видит целевой объект
COOKIE_SURVIVOR_FIXING = 3      # за попытку починки
COOKIE_SURVIVOR_FINISH = 6      # за завершение генератора
COOKIE_HUNTER_SPOT = 1          # за обнаружение выживших
COOKIE_HUNTER_CHASE = 2         # за сокращение дистанции во время погони
# «Чувство» генератора: базовое малое уменьшение и прирост по близости
COOKIE_SURVIVOR_SENSE_DECAY = 0.05
COOKIE_SURVIVOR_SENSE_GAIN = 0.6
COOKIE_SURVIVOR_SENSE_MAX_DIST = 600
COOKIE_HUNTER_DISTANCE_DECAY = 0.2
COOKIE_HUNTER_KILL_REWARD = 30
COOKIE_HUNTER_APPROACH_GAIN = 0.6

# Obstacle Field parameters (поле препятствий)
OBSTACLE_FIELD_ENABLED = True  # Включить поле препятствий
OBSTACLE_WALL_SIZE = 30  # Размер временной стенки
OBSTACLE_SPAWN_DISTANCE = 80  # Расстояние перед агентом, на котором создается стенка
OBSTACLE_WALL_DURATION = 180  # Длительность существования стенки в шагах (примерно 12 секунд при 15 FPS)
OBSTACLE_COOLDOWN = 300  # Откат между созданием стенок для одного агента (примерно 20 секунд)