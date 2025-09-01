from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Optional


class Env(ABC):
    """
    强化学习环境接口基类
    定义所有环境必须实现的标准方法
    """

    @abstractmethod
    def reset(self) -> Any:
        """
        重置环境到初始状态
        返回:
            observation: 初始观察值
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        在环境中执行一个动作
        参数:
            action: 要执行的动作
        返回:
            observation: 执行动作后的观察值
            reward: 获得的奖励
            done: 是否结束（True/False）
            info: 附加信息字典
        """
        pass

    @abstractmethod
    def render(self, mode: str = 'human') -> Optional[Any]:
        """
        渲染当前环境状态
        参数:
            mode: 渲染模式 ('human', 'rgb_array', 等)
        返回:
            取决于渲染模式（图像数据或None）
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """关闭环境，释放资源"""
        pass

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """返回动作空间定义"""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """返回观察空间定义"""
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict:
        """返回环境元数据"""
        pass

    @property
    @abstractmethod
    def reward_range(self) -> Tuple[float, float]:
        """返回奖励范围 (min, max)"""
        pass
