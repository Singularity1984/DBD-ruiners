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

# DQN parameters (управляются внутри DQNAgent)
EPISODES = 100000

# Capture parameters
CAPTURE_RADIUS = 40
CAPTURE_HOLD_STEPS = 12

# Stuck avoidance
STUCK_THRESHOLD = 20
ESCAPE_BOOST = 0.3
POSITION_MEMORY = 50

# Agent parameters
SURVIVOR_SPEED = 3.0
HUNTER_SPEED = 2.8
SURVIVOR_RADIUS = 11
HUNTER_RADIUS = 14
SURVIVOR_VISION_RADIUS = 280
HUNTER_VISION_RADIUS = 200
SURVIVOR_VISION_ANGLE = 150
HUNTER_VISION_ANGLE = 120

# Generation parameters
NUM_SURVIVORS = (4, 4)  # 4 убегающих агента
NUM_HUNTERS = 1
NUM_GENERATORS = (5, 5)  # Всегда 5 генераторов
NUM_EXITS = (2, 2)  # Всегда 2 выхода

# Simulation settings
ENABLE_HUNTER = True  # 1 догоняющий агент
DEFAULT_SIMULATION_SPEED = 1.0
FPS = 15

# Async settings
ASYNC_BATCH_SIZE = 5  # Количество агентов для параллельного обновления
ASYNC_SLEEP_TIME = 0.001  # Время сна между батчами для предотвращения блокировки
