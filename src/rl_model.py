import json
import os.path
import pickle
import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple

ACTIONS = ["up", "down", "left", "right", "zoom_in", "zoom_out"]
MODEL_DIR = '../models'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

class ReinforcementLearning(ABC):

    @abstractmethod
    def select_action(self, i, j, scale_bin, actions) -> str:
        pass

    @abstractmethod
    def update(self, s, a, r, s2):
        pass

    @abstractmethod
    def save(self, filename: str):
        pass

    @abstractmethod
    def load(self, filename: str):
        pass

    @abstractmethod
    def clear(self):
        pass


class RLQtableModel(ReinforcementLearning):

    def __init__(self, eps=0.15, alpha=0.5, gamma=0.9, save=False, load: bool = False,
                 model: str = 'qtable.pkl'):
        self._save = save
        self.model: Dict[Tuple[int, int, int, str], float] = {}  # (i,j,scale_bin,action) -> Q
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        if load and os.path.exists(model):
            self.load(model)

    def _key(self, i, j, scale_bin, a, avg_conf: float):
        return (i, j, scale_bin, a, avg_conf < 0.3)

    def select_action(self, i, j, scale_bin, actions, avg_conf) -> str:
        if not actions:
            return ''
        # ε-greedy
        if random.random() < self.eps:
            return random.choice(actions)
        # Select an action with max Q, if not seen, randomly choice one.
        best_a, best_q = None, -1e9
        for a in actions:
            q = self.model.get(self._key(i, j, scale_bin, a, avg_conf), 0.0)
            if q > best_q:
                best_q, best_a = q, a
        return best_a or random.choice(actions)

    def update(self, status, a, r, status_new):
        if not self._save or not status_new[3]:
            return
        (i, j, scale_bin, actions, avg_conf) = status
        (i2, j2, scale_bin2, actions2, avg_conf2) = status_new
        key = self._key(i, j, scale_bin, a, avg_conf)
        q = self.model.get(key, 0.0)
        q_vals = [self.model.get(self._key(i2, j2, scale_bin2, ap, avg_conf2), 0.0) for ap in actions2]
        max_next = max(q_vals) if q_vals else 0
        # New QValue = QValue + alpha * (reward + gamma * max_next - Q)
        new_q = q + self.alpha * (r + self.gamma * max_next - q)
        self.model[key] = new_q

    def save(self, filename: str = 'qtable.pkl'):
        """
        save model
        :param filename: filename
        """
        full_path = os.path.join(MODEL_DIR, filename)
        if filename.endswith('.pkl'):
            with open(full_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'eps': self.eps,
                    'alpha': self.alpha,
                    'gamma': self.gamma,
                }, f)
        elif filename.endswith('.json'):
            # Save Q table in JSON format
            qtable_serializable = {
                f"{k[0]},{k[1]},{k[2]},{k[3]}": v for k, v in self.model.items()
            }
            data = {
                'Q': qtable_serializable,
                'eps': self.eps,
                'alpha': self.alpha,
                'gamma': self.gamma,
            }
            with open(full_path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError("format must be 'pickle' or 'json'")
        print(f"Q table was saved to {full_path}.")

    def load(self, filename: str = 'qtable.pkl'):
        """
        load a model from file
        :param filename: filename
        :return:
        """
        full_path = os.path.join(MODEL_DIR, filename)
        try:
            if filename.endswith('.pkl'):
                with open(full_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.eps = data.get('eps', self.eps)
                    self.alpha = data.get('alpha', self.alpha)
                    self.gamma = data.get('gamma', self.gamma)
            elif filename.endswith('.json'):
                with open(full_path, 'r') as f:
                    data = json.load(f)
                    # Convert json data to an object
                    self.model = {}
                    for key_str, value in data['model'].items():
                        parts = key_str.split(',')
                        key_tuple = (int(parts[0]), int(parts[1]), int(parts[2]), parts[3])
                        self.model[key_tuple] = value
                    self.eps = data.get('eps', self.eps)
                    self.alpha = data.get('alpha', self.alpha)
                    self.gamma = data.get('gamma', self.gamma)
            print(f"Load a model from {full_path}.")

        except FileNotFoundError:
            print(f"Load a model failed: {full_path} not exit.")
        except Exception as e:
            print(f"Load a model failed, message: {e}")

    def clear(self):
        """Clear Q table"""
        self.model.clear()
        print("Cleared Q table.")
