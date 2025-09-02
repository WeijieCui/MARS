import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from mars.agent import ACTIONS
from mars.detector import BaseDetector


# ----------------------------
# 环境：维护窗口、栅格、与检测器交互
# ----------------------------
class SearchEnv:
    def __init__(self, image_bgr: np.ndarray, detector: BaseDetector, target: str = '', grid_n=10):
        self.img = image_bgr
        self.detector = detector
        self.target = target
        self.H, self.W = image_bgr.shape[:2]
        self.grid_n = grid_n
        self.grid = np.zeros((grid_n, grid_n), dtype=float)  # 0/ -1 / (0,1]
        self.cx = self.W // 2
        self.cy = self.H // 2
        self.scale_levels = [0.18, 0.25, 0.35, 0.5, 0.7, 1.0]  # 缩放档位
        self.scale_idx = len(self.scale_levels) // 2  # 对应 ~0.7
        # 初始窗口：图像较大区域
        base = self.scale_levels[self.scale_idx]
        self.win_w = int(self.W * base)
        self.win_h = int(self.H * base)
        self._fit_window()
        self.actions = [*ACTIONS]

    def view(self):
        x1, y1, x2, y2 = self._fit_window()
        return self.img[y1:y2, x1:x2]

    def _fit_window(self):
        # 根据中心与比例，确定窗口并裁剪在图像范围内
        ratio = self.scale_levels[self.scale_idx]
        self.win_w = max(24, int(self.W * ratio))
        self.win_h = max(24, int(self.H * ratio))
        x1 = int(self.cx - self.win_w // 2)
        y1 = int(self.cy - self.win_h // 2)
        x2 = x1 + self.win_w
        y2 = y1 + self.win_h
        x1 = max(0, min(x1, self.W - 1))
        y1 = max(0, min(y1, self.H - 1))
        x2 = max(1, min(x2, self.W))
        y2 = max(1, min(y2, self.H))
        # 更新中心以确保窗口在图内
        self.cx = (x1 + x2) // 2
        self.cy = (y1 + y2) // 2
        return x1, y1, x2, y2

    def _cell_of(self, x, y) -> Tuple[int, int]:
        """返回像素坐标所在的 10x10 单元索引 (i,j)"""
        i = int(y / self.H * self.grid_n)
        j = int(x / self.W * self.grid_n)
        i = max(0, min(self.grid_n - 1, i))
        j = max(0, min(self.grid_n - 1, j))
        return i, j

    def step(self, action: str) -> (Tuple)[
        Tuple[int, int, int],
        float,
        List[Dict[str, Any]],
        Tuple[int, int, int],
        np.ndarray]:
        """
        执行动作 -> 检测 -> 更新栅格
        返回: s(当前i,j,scale_bin), r(奖励), obbs(此次检测的结果), s2(新状态)
        """
        # 当前状态（以窗口中心映射到网格）
        i, j = self._cell_of(self.cx, self.cy)
        s = (i, j, self.scale_idx)

        # 1) 在当前窗口执行检测
        x1, y1, x2, y2 = self._fit_window()
        crop = self.img[y1:y2, x1:x2]
        obbs, best_conf = self.detector.infer_obb(crop, self.target)
        fix_obbs(obbs, x1, y1)
        # 更新探索矩阵：以窗口中心所在格为记录点
        if best_conf > 0:
            self.grid[i, j] = max(self.grid[i, j], best_conf)  # 命中：写入更高置信度
        else:
            if self.grid[i, j] == 0:
                self.grid[i, j] = -1  # 首次访问且未命中

        # 简易奖励设计：命中加奖励，越高越好；未命中小惩罚；重复访问略惩罚
        r = 0.0
        if best_conf > 0:
            r += 1.0 + best_conf  # 命中更鼓励
            if self.scale_idx < len(self.scale_levels) - 1:
                r += 0.1  # 放大可获得更细粒度
        else:
            r -= 0.2
            if self.grid[i, j] != 0 and self.grid[i, j] != -1:
                r -= 0.1  # 理论不会触发；以防重复覆盖

        # 2) 状态转移：根据动作移动/缩放窗口
        step_px = max(20, self.scale_levels[self.scale_idx] * min(self.W, self.H) / 2)
        # step_px = max(12, min(self.W, self.H) // 12)
        if action == "up":
            self.cy = self.cy - step_px
            if self.cy <= 0:
                self.cy = 0
                self.actions.remove('up')
            elif 'down' not in self.actions:
                self.actions.append('down')
        elif action == "down":
            self.cy = self.cy + step_px
            if self.cy >= self.H - 1:
                self.cy = self.H - 1
                self.actions.remove('down')
            elif 'up' not in self.actions:
                self.actions.append('up')
        elif action == "left":
            self.cx = self.cx - step_px
            if self.cx <= 0:
                self.cx = 0
                self.actions.remove('left')
            elif 'right' not in self.actions:
                self.actions.append('right')
        elif action == "right":
            self.cx = self.cx + step_px
            if self.cx >= self.W - 1:
                self.cx = self.W - 1
                self.actions.remove('right')
            elif 'left' not in self.actions:
                self.actions.append('left')
        elif action == "zoom_in":
            self.scale_idx = self.scale_idx + 1
            if self.scale_idx >= len(self.scale_levels) - 1:
                self.scale_idx = len(self.scale_levels) - 1
                self.actions.remove('zoom_in')
            elif 'zoom_out' not in self.actions:
                self.actions.append('zoom_out')
        elif action == "zoom_out":
            self.scale_idx = max(self.scale_idx - 1, 0)
            if self.scale_idx <= 0:
                self.scale_idx = 0
                self.actions.remove('zoom_out')
            elif 'zoom_in' not in self.actions:
                self.actions.append('zoom_in')

        # 新状态
        i2, j2 = self._cell_of(self.cx, self.cy)
        s2 = (i2, j2, self.scale_idx, self.actions)
        return s, r, obbs, s2, crop


def fix_obbs(obbs: List[Dict[str, Any]], x: float, y: float):
    for obb in obbs:
        obb['cx'] += x
        obb['cy'] += y
