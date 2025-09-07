import math
import numpy as np
from typing import List, Dict, Any, Tuple

from mars.detector import BaseDetector
from utils import merge_bounding_box

DEFAULT_SCALE_LEVELS = [0.063, 0.125, 0.25, 0.5, 1.0]


# ----------------------------
# Environment
# ----------------------------
class SearchEnv:
    def __init__(self, grid_n=10):
        self.H, self.W = 0, 0
        self.img = None
        self.target = -1
        self.references = None
        self.scale_levels = None
        self.scale_idx = None
        self.multi_grids = None
        # window sizes
        self.win_w, self.win_h = 0, 0
        self.actions = ['zoom_in']
        self.found_objects = []
        self.new_found_objects = []

    def get_grid(self):
        return self.multi_grids[self.scale_idx]

    def border(self):
        x1, y1, x2, y2 = self._fit_window()
        return np.array([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])

    def set_image(self, image_bgr: np.ndarray):
        """
        Set an image, update relevant status
        :param image_bgr: image arrays in BGR format
        """
        self.img = image_bgr
        self.H, self.W = image_bgr.shape[:2]
        self.cx = self.W // 2
        self.cy = self.H // 2
        self.scale_levels = DEFAULT_SCALE_LEVELS[
                            max(0, len(DEFAULT_SCALE_LEVELS) - int(math.sqrt(min(image_bgr.shape[:2]) / 300)) - 1):]
        self.scale_idx = len(self.scale_levels) - 1  # 1
        self.multi_grids = {
            i: np.zeros((2 ** (len(self.scale_levels) - i - 1), 2 ** (len(self.scale_levels) - i - 1)), dtype=float)
            for i in range(len(self.scale_levels))}
        base = self.scale_levels[self.scale_idx]
        self.win_w = int(self.W * base)
        self.win_h = int(self.H * base)
        self._fit_window()

    def set_detector(self, detector: BaseDetector):
        self.detector = detector

    def set_target(self, target: str = '-1'):
        self.target = int(target)

    def set_references(self, references: str = ''):
        self.references = references

    def view(self):
        x1, y1, x2, y2 = self._fit_window()
        return self.img[y1:y2, x1:x2]

    def _fit_window(self):
        """
        Fit window to available area
        :return: x1, y1, x2, y2
        """
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
        # Update the center
        self.cx = (x1 + x2) // 2
        self.cy = (y1 + y2) // 2
        return x1, y1, x2, y2

    def _get_grid_size(self):
        return 2 ** (len(self.multi_grids) - self.scale_idx - 1)

    def _cell_of_zoom_out(self, x, y) -> Tuple[int, int]:
        """return new indexes of zoom out"""
        if self.scale_idx >= len(self.multi_grids) - 1:
            raise Exception
        grid_size = self._get_grid_size() // 2
        i = int(y / self.H * grid_size)
        j = int(x / self.W * grid_size)
        i = max(0, min(grid_size - 1, i))
        j = max(0, min(grid_size - 1, j))
        return i, j

    def _cell_of_zoom_in(self, x, y) -> Tuple[int, int]:
        """return new indexes of zoom in"""
        if self.scale_idx <= 0:
            raise Exception
        grid_size = self._get_grid_size() * 2
        i = int(y / self.H * grid_size)
        j = int(x / self.W * grid_size)
        i = max(0, min(grid_size - 1, i))
        j = max(0, min(grid_size - 1, j))
        return i, j

    def _cell_of(self, x, y) -> Tuple[int, int]:
        """calculate current indexes"""
        grid_size = self._get_grid_size()
        i = int(y / self.H * grid_size)
        j = int(x / self.W * grid_size)
        i = max(0, min(grid_size - 1, i))
        j = max(0, min(grid_size - 1, j))
        return i, j

    def reset(self):
        self.found_objects = []
        self.new_found_objects = []
        self.set_image(self.img)
        self.actions = ['zoom_in']
        i, j = self._cell_of(self.cx, self.cy)
        x1, y1, x2, y2 = self._fit_window()
        crop = self.img[y1:y2, x1:x2]
        reward = 0.0
        return (i, j, self.scale_idx, self.actions), reward, self.found_objects, self.new_found_objects, crop

    def get_status(self):
        i, j = self._cell_of(self.cx, self.cy)
        return (i, j, self.scale_idx, self.actions)

    def step(self, action: str) -> (Tuple)[
        Tuple[int, int, int, List[str]],
        float,
        List[Dict[str, Any]],
        Tuple[int, int, int],
        np.ndarray]:
        """
        执行动作 -> 检测 -> 更新栅格
        返回: s(当前i,j,scale_bin), r(奖励), obbs(此次检测的结果), status_new(新状态)
        """
        # 当前状态（以窗口中心映射到网格）
        assert action in self.actions
        # 2) 状态转移：根据动作移动/缩放窗口
        step_px = max(20, self.scale_levels[self.scale_idx] * self.W)
        step_py = max(20, self.scale_levels[self.scale_idx] * self.H)
        if action == "up":
            self.cy = self.cy - step_py
            if self.cy <= 0:
                self.cy = 0
        elif action == "down":
            self.cy = self.cy + step_py
            if self.cy >= self.H - 1:
                self.cy = self.H - 1
        elif action == "left":
            self.cx = self.cx - step_px
            if self.cx <= 0:
                self.cx = 0
        elif action == "right":
            self.cx = self.cx + step_px
            if self.cx >= self.W - 1:
                self.cx = self.W - 1
        elif action == "zoom_out":
            self.scale_idx += 1
            if self.scale_idx >= len(self.scale_levels) - 1:
                self.scale_idx = len(self.scale_levels) - 1
        elif action == "zoom_in":
            self.scale_idx -= 1
            if self.scale_idx <= 0:
                self.scale_idx = 0
        self.refresh_actions()
        # 新状态
        i, j = self._cell_of(self.cx, self.cy)
        # 1) 在当前窗口执行检测
        x1, y1, x2, y2 = self._fit_window()
        crop = self.img[y1:y2, x1:x2]
        obbs, _ = self.detector.infer_obb(crop, self.target)
        fix_obbs(obbs, x1, y1)
        self.new_found_objects = merge_bounding_box(self.found_objects, obbs)
        self.found_objects.extend(self.new_found_objects)
        scores = [obb['score'] for obb in self.new_found_objects]
        best_conf = max(scores) if scores else -1
        # 更新探索矩阵：以窗口中心所在格为记录点
        if best_conf > 0:
            self.multi_grids[self.scale_idx][i, j] = max(self.multi_grids[self.scale_idx][i, j],
                                                         best_conf)  # 命中：写入更高置信度
        else:
            if self.multi_grids[self.scale_idx][i, j] == 0 or len(self.new_found_objects) == 0:
                self.multi_grids[self.scale_idx][i, j] = -1  # 首次访问且未命中
        # 简易奖励设计：命中加奖励，越高越好；未命中小惩罚；重复访问略惩罚
        reward = 0.0
        if best_conf > 0:
            reward += 1.0 + best_conf  # 命中更鼓励
            if self.scale_idx < len(self.scale_levels) - 1:
                reward += 0.1  # Fine grants
        else:
            reward -= 0.2
            if self.multi_grids[self.scale_idx][i, j] != 0 and self.multi_grids[self.scale_idx][i, j] != -1:
                reward -= 0.1  # In case of overwrite
        return (i, j, self.scale_idx, self.actions), reward, self.found_objects, self.new_found_objects, crop

    def refresh_actions(self):
        idx_y, idx_x = self._cell_of(self.cx, self.cy)
        grid_size = self._get_grid_size()
        grid = self.multi_grids[self.scale_idx]
        if 'up' in self.actions:
            if idx_y <= 0 or grid[idx_y - 1, idx_x] != 0 or self.cy <= 0:
                self.actions.remove('up')
        elif idx_y > 0 and grid[idx_y - 1, idx_x] == 0 and self.cy > 0:
            self.actions.append('up')
        if 'down' in self.actions:
            if idx_y >= grid_size - 1 or grid[idx_y + 1, idx_x] != 0 or self.cy >= self.H - 1:
                self.actions.remove('down')
        elif idx_y < grid_size - 1 and grid[idx_y + 1, idx_x] == 0 and self.cy < self.H - 1:
            self.actions.append('down')
        if 'left' in self.actions:
            if idx_x <= 0 or grid[idx_y, idx_x - 1] != 0 or self.cx <= 0:
                self.actions.remove('left')
        elif idx_x > 0 and grid[idx_y, idx_x - 1] == 0 and self.cx > 0:
            self.actions.append('left')
        if 'right' in self.actions:
            if (idx_x >= grid_size - 1) or (grid[idx_y, idx_x + 1] != 0) or (self.cx >= self.W - 1):
                self.actions.remove('right')
        elif idx_x < grid_size - 1 and grid[idx_y, idx_x + 1] == 0 and self.cx < self.W - 1:
            self.actions.append('right')
        if 'zoom_out' in self.actions:
            if self.scale_idx < len(self.multi_grids) - 1:
                iu, ju = self._cell_of_zoom_out(self.cx, self.cy)
                grid_out = self.multi_grids[self.scale_idx + 1]
                if grid_out[iu, ju] != 0:
                    self.actions.remove('zoom_out')
            else:
                self.actions.remove('zoom_out')
        else:
            if self.scale_idx < len(self.multi_grids) - 1:
                iu, ju = self._cell_of_zoom_out(self.cx, self.cy)
                grid_out = self.multi_grids[self.scale_idx + 1]
                if grid_out[iu, ju] == 0:
                    self.actions.append('zoom_out')
        if 'zoom_in' in self.actions:
            if self.scale_idx > 0:
                iu, ju = self._cell_of_zoom_in(self.cx, self.cy)
                grid_in = self.multi_grids[self.scale_idx - 1]
                if grid_in[iu, ju] != 0:
                    self.actions.remove('zoom_in')
            else:
                self.actions.remove('zoom_in')
        else:
            if self.scale_idx > 0:
                iu, ju = self._cell_of_zoom_in(self.cx, self.cy)
                grid_in = self.multi_grids[self.scale_idx - 1]
                if grid_in[iu, ju] == 0:
                    self.actions.append('zoom_in')


def fix_obbs(obbs: List[Dict[str, Any]], x: float, y: float):
    for obb in obbs:
        obb['cx'] += x
        obb['cy'] += y
