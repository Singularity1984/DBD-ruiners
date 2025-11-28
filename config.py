# Конфигурационные параметры
WIDTH, HEIGHT = 1200, 800
BACKGROUND = (15, 15, 20)
TEXT_COLOR = (220, 220, 220)
UI_BG = (30, 30, 40, 200)

# Цвета агентов
SURVIVOR_COLOR = (70, 130, 200)
HUNTER_COLOR = (200, 50, 50)

# Цвета предметов (ДОБАВЛЕНО БОЛЬШЕ ЦВЕТОВ)
GENERATOR_COLOR = (150, 150, 150)
GENERATOR_FIXED_COLOR = (0, 200, 0)
GENERATOR_PROGRESS_COLOR = (0, 255, 0)

EXIT_COLOR = (100, 200, 100)
EXIT_ACTIVE_COLOR = (0, 255, 0)

WALL_COLOR = (80, 80, 100)
WALL_BORDER_COLOR = (60, 60, 80)

HOOK_COLOR = (139, 69, 19)  # Коричневый
HOOK_OCCUPIED_COLOR = (200, 0, 0)
HOOK_PROGRESS_COLOR = (220, 20, 60)  # Красный
HOOK_PROGRESS_WARNING_COLOR = (255, 140, 0)  # Оранжевый

LOCKER_COLOR = (70, 70, 80)
LOCKER_OCCUPIED_COLOR = (100, 0, 0)
LOCKER_BORDER_COLOR = (40, 40, 50)

TOTEM_COLOR = (200, 200, 100)  # Желтоватый
TOTEM_HEX_COLOR = (220, 20, 60)  # Красный
TOTEM_CLEANSED_COLOR = (100, 100, 100)  # Серый

# Цвета состояний агентов
SURVIVOR_INJURED_COLOR = (255, 100, 100)
SURVIVOR_DYING_COLOR = (200, 0, 0)
SURVIVOR_FIXING_COLOR = (255, 255, 0)
SURVIVOR_CARRIED_COLOR = (150, 150, 150)
SURVIVOR_IN_LOCKER_COLOR = (100, 100, 200)
SURVIVOR_ESCAPED_COLOR = (0, 200, 0)
SURVIVOR_CAUGHT_COLOR = (100, 100, 100)

# Цвета надписей
LABEL_COLOR = (255, 255, 255)
LABEL_BACKGROUND = (0, 0, 0, 180)  # Полупрозрачный черный

# Зрение
SURVIVOR_VISION_RADIUS = 250
HUNTER_VISION_RADIUS = 300
SURVIVOR_VISION_ANGLE = 120
HUNTER_VISION_ANGLE = 90

# Обучение
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
EPISODES = 100000

# Q-Table
MAX_Q_TABLE_SIZE = 10000

# Игровая логика - КОЛИЧЕСТВО ОБЪЕКТОВ (ВСЁ НАСТРАИВАЕТСЯ!)
NUM_SURVIVORS = (3, 4)           # Количество выживших (мин, макс)
NUM_GENERATORS = (4, 5)          # Количество генераторов (мин, макс)
NUM_EXITS = (1, 2)               # Количество выходов (мин, макс)
NUM_WALLS = (8, 12)              # Количество стен (мин, макс)
NUM_HOOKS = (4, 6)               # Количество крюков (мин, макс) - ДОБАВЛЕНО
NUM_LOCKERS = (6, 8)             # Количество шкафов (мин, макс) - ДОБАВЛЕНО
NUM_TOTEMS = (4, 5)              # Количество тотемов (мин, макс) - ДОБАВЛЕНО
NUM_HEX_TOTEMS = (1, 2)          # Количество проклятых тотемов (мин, макс) - ДОБАВЛЕНО

# Захват
CAPTURE_RADIUS = 30
CAPTURE_HOLD_STEPS = 60

# Параметры движения
STUCK_THRESHOLD = 30
POSITION_MEMORY = 50
ESCAPE_BOOST = 0.3

# Визуализация
FPS = 60
DEFAULT_SIMULATION_SPEED = 1.0
MAX_STEPS_PER_FRAME = 1000
DISABLE_VISUALIZATION_ABOVE = 500

# Надписи (НОВАЯ СЕКЦИЯ)
SHOW_LABELS_BELOW_SPEED = 500  # Показывать надписи только при скорости ниже этого значения
LABEL_FONT_SIZE = 14

# Файлы
SURVIVORS_SAVE_FILE = "survivors_model.pkl"
HUNTER_SAVE_FILE = "hunter_model.pkl"

# Охотник
ENABLE_HUNTER = True

# Сколько эпизодов хранит Q-Table
PRUNE_EVERY = 500

# ДОБАВЛЕНО: Скорости агентов
SURVIVOR_SPEED = 3
HUNTER_SPEED = 2.8

# ДОБАВЛЕНО: Радиусы агентов
SURVIVOR_RADIUS = 12
HUNTER_RADIUS = 15

# ДОБАВЛЕНО: Бонусы от тотемов для охотника
HEX_TOTEM_SPEED_BONUS = 0.2  # +20% скорости за каждый активный проклятый тотем
HEX_TOTEM_VISION_BONUS = 50   # +50 к радиусу зрения за каждый активный проклятый тотем
NORMAL_TOTEM_SPEED_BONUS = 0.05  # +5% скорости за каждый активный обычный тотем

# ДОБАВЛЕНО: Настройки крюков
HOOK_PROGRESS_SPEED = 0.3     # Скорость прогресса на крюке (чем больше - тем быстрее умирают)
HOOK_STAGE_DURATION = 100     # Длительность каждой стадии на крюке (в процентах)
