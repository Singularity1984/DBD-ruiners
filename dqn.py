import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from config import *


class DQN(nn.Module):
    """Улучшенная Deep Q-Network для обучения агентов"""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQN, self).__init__()
        # Улучшенная архитектура с Layer Normalization (работает с batch size = 1)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.1)
        
        # Инициализация весов
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Инициализация весов для лучшей сходимости"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Обеспечиваем правильную размерность
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """Оптимизированный буфер для хранения опыта"""
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Добавление опыта в буфер"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Выборка случайного батча"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Улучшенный агент с DQN обучением"""
    
    def __init__(self, state_size, action_size, is_hunter=False):
        self.state_size = state_size
        self.action_size = action_size
        self.is_hunter = is_hunter
        
        # Оптимизированные параметры обучения
        self.learning_rate = 0.0005 if is_hunter else 0.001
        self.gamma = 0.99  # Увеличенный discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 128  # Увеличенный batch size
        self.memory = ReplayBuffer(50000)  # Увеличенный буфер
        self.update_target_every = 200  # Реже обновляем target network
        
        # Нейронные сети
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, action_size, hidden_size=256).to(self.device)
        self.target_network = DQN(state_size, action_size, hidden_size=256).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # Синхронизация target network
        self.update_target_network()
        self.steps = 0
        self.training_steps = 0
        
    def update_target_network(self):
        """Обновление target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Сохранение опыта в буфер"""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        """Выбор действия с epsilon-greedy стратегией"""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.cpu().data.numpy().argmax()
    
    def replay(self):
        """Улучшенное обучение на батче с gradient clipping"""
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Текущие Q-значения
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Следующие Q-значения из target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Вычисление loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Обучение
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Обновление epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Обновление target network
        self.steps += 1
        self.training_steps += 1
        if self.training_steps % self.update_target_every == 0:
            self.update_target_network()
        
        return loss.item()
    
    def save(self, filepath):
        """Сохранение модели"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'training_steps': self.training_steps
        }, filepath)
    
    def load(self, filepath):
        """Загрузка модели с обработкой несовместимости архитектуры"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Пытаемся загрузить состояние
            try:
                self.q_network.load_state_dict(checkpoint['q_network'], strict=False)
                self.target_network.load_state_dict(checkpoint['q_network'], strict=False)
            except Exception as e:
                # Если архитектура не совместима, загружаем только совместимые части
                print(f"Предупреждение: архитектура модели изменилась. Загружаются только совместимые параметры.")
                try:
                    old_state = checkpoint['q_network']
                    new_state = self.q_network.state_dict()
                    
                    # Загружаем только совместимые слои
                    for key in new_state.keys():
                        if key in old_state:
                            if new_state[key].shape == old_state[key].shape:
                                new_state[key] = old_state[key]
                            else:
                                print(f"Пропущен слой {key} из-за несовместимости размеров")
                    
                    self.q_network.load_state_dict(new_state)
                    self.target_network.load_state_dict(new_state)
                except Exception as e2:
                    print(f"Не удалось загрузить модель: {e2}. Начинаем обучение с нуля.")
                    return False
            
            # Загружаем остальные параметры если они есть
            if 'optimizer' in checkpoint:
                try:
                    self.optimizer.load_state_dict(checkpoint['optimizer'])
                except:
                    pass  # Игнорируем ошибки оптимизатора
            
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps = checkpoint.get('steps', 0)
            self.training_steps = checkpoint.get('training_steps', 0)
            self.update_target_network()
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
