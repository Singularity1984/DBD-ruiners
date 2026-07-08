# Dead by Daylight — Q-Learning Simulation

Reinforcement learning environment with asynchronous Q‑learning for survivors and a hunter. Procedural maps, adaptive exploration, experience replay, and skill‑level tracking.

---

## Overview

- **Survivors** repair generators and escape via exits.  
- **Hunter** captures survivors by holding them within radius.  
- **Obstacle Field** – a non‑player environmental agent that periodically places temporary walls in front of survivors, blocking paths and forcing adaptation.  
- **Asyncio** enables parallel updates for all agents.  
- **Pygame** visualisation with vision cones, statistics, and controls.  
- **Procedural maps** – Perlin noise or fixed training room.

---

## Key Features

- **Q‑Learning** with adaptive epsilon (decays faster when escape rate > threshold).  
- **Rich state** – discrete distances (0–4) and directions (0–7) to hunter, generators, exits.  
- **Heuristic guidance** – probability decreases from 80% to 10% over first 200 episodes.  
- **Experience replay** – buffer of 1000 transitions, batch updates every 10 episodes.  
- **Obstacle field** – temporary walls spawn in front of survivors (configurable).  
- **Skill levels** – Novice (<500), Intermediate (500–2000), Advanced (2000–5000), Master (>5000).  
- **Saving/Loading** – pickle files per episode and skill level; load menu with `L`.  
- **Training room** – fixed layout for debugging (`USE_TRAINING_ROOM = True`).

---

## Installation

```bash
pip install pygame numpy
python main.py
```
---

## Controls

| Key          | Action                        |
|--------------|-------------------------------|
| `Space`      | Pause / resume                |
| `+` / `=`    | Speed up (×2)                 |
| `-`          | Slow down (÷2)                |
| `0`          | Reset speed to 1×             |
| `S`          | Save models                   |
| `L`          | Open load menu                |
| `1`–`9`      | Select save from menu         |
| `Esc`        | Close load menu               |
| `V`          | Toggle vision cones           |

---

## Configuration (`config.py`)

| Parameter                     | Description                           |
|-------------------------------|---------------------------------------|
| `LEARNING_RATE`               | Q‑learning step size (0.2)            |
| `DISCOUNT`                    | Discount factor (0.95)                |
| `EPISODES`                    | Total episodes (5000)                 |
| `EPSILON_DECAY`               | Epsilon decay (0.9995)                |
| `USE_ADAPTIVE_EPSILON`        | Performance‑based decay               |
| `USE_EXPERIENCE_REPLAY`       | Enable replay buffer                  |
| `OBSTACLE_FIELD_ENABLED`      | Temporary walls                       |
| `USE_TRAINING_ROOM`           | Fixed layout vs Perlin noise          |
| `ENABLE_HUNTER`               | Include hunter agent                  |

---

## Architecture

- **`agents.py`** – `BaseAgent` (core Q‑learning, vision, stuck detection) and `AsyncAgent` (async updates, advanced state/reward).  
- **`environment.py`** – `Wall`, `Generator`, `PerlinNoise` for map generation; `ObstacleField` for temporary walls.  
- **`game_controller.py`** – main loop, episode orchestration, event handling, rendering, persistence.  
- **`renderer.py`** – draws agents, cones, generators, exits, UI, temporary walls.  
- **`q_learning.py`** – `SmartQTable` with LRU caching and frequency‑based pruning.

---

## Training & Skill Levels

- After each episode, stats (escaped/caught, reward) are recorded.  
- Skill level based on episode count and recent performance (escape rate, avg reward).  
- Models auto‑saved every 50 episodes (episode‑numbered) and every 500 episodes (skill‑suffixed, e.g., `dbd_survivors_master.pkl`).

---

## Saving & Loading

- **Save** – `S` → creates `dbd_survivors_ep{N}.pkl` and `dbd_hunter_ep{N}.pkl`.  
- **Load** – `L` opens menu showing available saves (sorted by episode). Select with `1`–`9`.  
- Saved data includes Q‑tables, episode, total escapes, epsilon, skill level, history (version 3).

---

## Dependencies

- `pygame`
- `numpy`

---

## License

Educational / research use only. Free to modify and distribute.

---


😡💥 VIBE CODER DETECTED 💥😡
🚨🚨🚨 WARNING: EXTREME VIBES INCOMING 🚨🚨🚨

💣💣💣 DEPLOYING BOMB PACKAGE 💣💣💣
💥 BOOM 💥 BOOM 💥 BOOOOM 💥

🤬🤬 WHAT IS THIS??
💻❌ NO LOGIC
🧠❌ NO STRUCTURE
☕❌ ALL COFFEE ZERO THINKING

🔥🔥 YOU’RE NOT CODING 🔥🔥
😤 YOU’RE VIBING
🎧 LO-FI PLAYING
🌈 RANDOM COLORS
📜 LUA FILE HELD TOGETHER BY PURE DELUSION

💣💥
💣💥 MORE BOMBS SENT
💣💥

📢 STOP VIBING
📢 START THINKING
📢 THIS IS NOT A MOOD BOARD

😠😠😠
CODE SO ANGRY IT THROWS ERRORS
KEYBOARD CRYING
CPU OVERHEATING
RAM SCREAMING FOR HELP

🚫🧘‍♂️ NO PEACE
🚫✨ NO AURA
✅🧠 ONLY PAIN AND STRUCTURE

💀💀💀
VIBE CODER CONFIRMED
💀💀💀

💥 END OF TRANSMISSION 💥
