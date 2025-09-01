import pygame
import numpy as np
import random
import sys
from enum import Enum


# Action enumeration
class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# Reinforcement Learning Environment Interface
class RLEnvironment:
    """Base class for reinforcement learning environments"""

    def reset(self) -> tuple:
        """Reset the environment to initial state"""
        pass

    def step(self, action: Action) -> tuple:
        """Take an action in the environment"""
        pass

    def render(self) -> None:
        """Render the current environment state"""
        pass

    def close(self) -> None:
        """Close the environment and release resources"""
        pass

    @property
    def action_space(self) -> list:
        """Get available actions"""
        pass

    @property
    def observation_space(self) -> tuple:
        """Get observation space dimensions"""
        pass


# Grid World Environment Implementation
class GridWorldEnv(RLEnvironment):
    """
    Grid World Environment with visualization
    Agent navigates from start to goal while avoiding obstacles
    """

    # Color definitions for visualization
    COLORS = {
        'background': (240, 240, 240),
        'grid': (200, 200, 200),
        'agent': (30, 144, 255),  # DodgerBlue
        'goal': (50, 205, 50),  # LimeGreen
        'obstacle': (169, 169, 169),  # DarkGray
        'text': (40, 40, 40),
        'visited': (173, 216, 230),  # LightBlue
        'start': (255, 165, 0)  # Orange
    }

    def __init__(self, size: int = 8, obstacle_density: float = 0.15):
        # Environment parameters
        self.size = size
        self.obstacle_density = obstacle_density
        self.cell_size = 60  # Pixel size of each grid cell
        self.window_size = size * self.cell_size + 250  # Extra space for info panel

        # Environment state
        self.start_pos = (0, 0)
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = None
        self.visited = set()  # Track visited positions
        self.steps = 0
        self.total_reward = 0
        self.max_steps = size * size * 2  # Maximum steps before episode ends

        # Define action space
        self._action_space = list(Action)

        # Define observation space (agent position)
        self._observation_space = (2,)

        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Grid World Environment")
        self.font = pygame.font.SysFont('Arial', 24)
        self.title_font = pygame.font.SysFont('Arial', 36, bold=True)

        # Reset environment
        self.reset()

    def reset(self) -> tuple:
        """Reset environment to initial state"""
        # Set start position
        self.agent_pos = self.start_pos

        # Set goal position (bottom-right corner)
        self.goal_pos = (self.size - 1, self.size - 1)

        # Generate obstacles
        self.obstacles = set()
        self.visited = {self.agent_pos}
        self.steps = 0
        self.total_reward = 0

        # Randomly place obstacles
        num_obstacles = int(self.size * self.size * self.obstacle_density)
        for _ in range(num_obstacles):
            while True:
                pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if (pos != self.agent_pos and
                        pos != self.goal_pos and
                        pos not in self.obstacles):
                    self.obstacles.add(pos)
                    break

        return self.agent_pos

    def step(self, action: Action) -> tuple:
        """Execute an action in the environment"""
        x, y = self.agent_pos
        new_x, new_y = x, y

        # Calculate new position based on action
        if action == Action.UP:
            new_x = max(0, x - 1)
        elif action == Action.RIGHT:
            new_y = min(self.size - 1, y + 1)
        elif action == Action.DOWN:
            new_x = min(self.size - 1, x + 1)
        elif action == Action.LEFT:
            new_y = max(0, y - 1)

        # Check for obstacle collision
        if (new_x, new_y) in self.obstacles:
            new_x, new_y = x, y  # Stay in current position if collision

        # Update agent position
        self.agent_pos = (new_x, new_y)
        self.visited.add(self.agent_pos)
        self.steps += 1

        # Calculate reward
        done = (self.agent_pos == self.goal_pos)
        time_out = (self.steps >= self.max_steps)

        # Reward structure:
        # - Reaching goal: +100
        # - Each step: -1
        # - Visiting new cell: +0.5
        # - Obstacle collision: -5
        # - Timeout: -10
        reward = 0

        if done:
            reward += 100
        elif time_out:
            reward -= 10
            done = True
        else:
            reward -= 1

            # Check if moved to a new cell
            if (new_x, new_y) != (x, y):
                # Check if we hit an obstacle (but didn't move)
                if (new_x, new_y) == (x, y) and (x, y) in self.obstacles:
                    reward -= 5
                # Reward for exploring new cells
                elif (new_x, new_y) not in self.visited:
                    reward += 0.5

        self.total_reward += reward

        # Information dictionary
        info = {
            'position': self.agent_pos,
            'distance': abs(new_x - self.goal_pos[0]) + abs(new_y - self.goal_pos[1]),
            'visited': len(self.visited),
            'obstacles': len(self.obstacles),
            'timeout': time_out
        }

        return self.agent_pos, reward, done, info

    def render(self) -> None:
        """Render the current environment state"""
        # Handle PyGame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        # Clear screen
        self.screen.fill(self.COLORS['background'])

        # Draw grid lines
        for i in range(self.size + 1):
            # Horizontal lines
            pygame.draw.line(
                self.screen, self.COLORS['grid'],
                (0, i * self.cell_size),
                (self.size * self.cell_size, i * self.cell_size),
                2
            )
            # Vertical lines
            pygame.draw.line(
                self.screen, self.COLORS['grid'],
                (i * self.cell_size, 0),
                (i * self.cell_size, self.size * self.cell_size),
                2
            )

        # Draw visited cells
        for pos in self.visited:
            if pos != self.agent_pos and pos != self.goal_pos and pos != self.start_pos:
                pygame.draw.rect(
                    self.screen, self.COLORS['visited'],
                    (pos[1] * self.cell_size + 2, pos[0] * self.cell_size + 2,
                     self.cell_size - 4, self.cell_size - 4)
                )

        # Draw obstacles
        for pos in self.obstacles:
            # Obstacle background
            pygame.draw.rect(
                self.screen, self.COLORS['obstacle'],
                (pos[1] * self.cell_size + 2, pos[0] * self.cell_size + 2,
                 self.cell_size - 4, self.cell_size - 4)
            )
            # Draw X mark on obstacle
            pygame.draw.line(
                self.screen, (100, 100, 100),
                (pos[1] * self.cell_size + 10, pos[0] * self.cell_size + 10),
                ((pos[1] + 1) * self.cell_size - 10, (pos[0] + 1) * self.cell_size - 10),
                4
            )
            pygame.draw.line(
                self.screen, (100, 100, 100),
                (pos[1] * self.cell_size + 10, (pos[0] + 1) * self.cell_size - 10),
                ((pos[1] + 1) * self.cell_size - 10, pos[0] * self.cell_size + 10),
                4
            )

        # Draw goal position
        pygame.draw.rect(
            self.screen, self.COLORS['goal'],
            (self.goal_pos[1] * self.cell_size + 2, self.goal_pos[0] * self.cell_size + 2,
             self.cell_size - 4, self.cell_size - 4)
        )

        # Draw start position
        pygame.draw.rect(
            self.screen, self.COLORS['start'],
            (self.start_pos[1] * self.cell_size + 2, self.start_pos[0] * self.cell_size + 2,
             self.cell_size - 4, self.cell_size - 4)
        )

        # Draw agent
        center_x = self.agent_pos[1] * self.cell_size + self.cell_size // 2
        center_y = self.agent_pos[0] * self.cell_size + self.cell_size // 2
        pygame.draw.circle(
            self.screen, self.COLORS['agent'],
            (center_x, center_y),
            self.cell_size // 3
        )

        # Draw agent eyes
        pygame.draw.circle(
            self.screen, (255, 255, 255),
            (center_x - self.cell_size // 10, center_y - self.cell_size // 10),
            self.cell_size // 10
        )
        pygame.draw.circle(
            self.screen, (0, 0, 0),
            (center_x - self.cell_size // 10, center_y - self.cell_size // 10),
            self.cell_size // 20
        )

        # Draw information panel
        info_x = self.size * self.cell_size + 20

        # Environment title
        title = self.title_font.render("Grid World Environment", True, self.COLORS['text'])
        self.screen.blit(title, (info_x, 20))

        # Status information
        info_text = [
            f"Steps: {self.steps}",
            f"Total Reward: {self.total_reward:.2f}",
            f"Agent Position: {self.agent_pos}",
            f"Goal Position: {self.goal_pos}",
            f"Visited Cells: {len(self.visited)}/{self.size * self.size}",
            f"Obstacles: {len(self.obstacles)}",
            f"Max Steps: {self.max_steps}",
            "",
            "Controls:",
            "↑ - Move Up",
            "→ - Move Right",
            "↓ - Move Down",
            "← - Move Left",
            "",
            "R - Reset Environment",
            "ESC - Quit"
        ]

        for i, text in enumerate(info_text):
            rendered = self.font.render(text, True, self.COLORS['text'])
            self.screen.blit(rendered, (info_x, 100 + i * 30))

        # Draw legend
        legend_y = self.size * self.cell_size - 180
        pygame.draw.rect(self.screen, self.COLORS['agent'], (info_x, legend_y, 20, 20))
        self.screen.blit(self.font.render("Agent", True, self.COLORS['text']), (info_x + 30, legend_y))

        pygame.draw.rect(self.screen, self.COLORS['goal'], (info_x, legend_y + 30, 20, 20))
        self.screen.blit(self.font.render("Goal", True, self.COLORS['text']), (info_x + 30, legend_y + 30))

        pygame.draw.rect(self.screen, self.COLORS['obstacle'], (info_x, legend_y + 60, 20, 20))
        self.screen.blit(self.font.render("Obstacle", True, self.COLORS['text']), (info_x + 30, legend_y + 60))

        pygame.draw.rect(self.screen, self.COLORS['visited'], (info_x, legend_y + 90, 20, 20))
        self.screen.blit(self.font.render("Visited Cell", True, self.COLORS['text']), (info_x + 30, legend_y + 90))

        pygame.draw.rect(self.screen, self.COLORS['start'], (info_x, legend_y + 120, 20, 20))
        self.screen.blit(self.font.render("Start Position", True, self.COLORS['text']), (info_x + 30, legend_y + 120))

        # Update display
        pygame.display.flip()
        pygame.time.delay(100)  # Control rendering speed

    def close(self) -> None:
        """Close the environment and release resources"""
        pygame.quit()

    @property
    def action_space(self) -> list:
        """Get available actions"""
        return self._action_space

    @property
    def observation_space(self) -> tuple:
        """Get observation space dimensions"""
        return self._observation_space


# Main function to run the environment
def main():
    # Create environment
    env = GridWorldEnv(size=8, obstacle_density=0.15)

    # Main game loop
    running = True
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle key presses
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key == pygame.K_UP:
                    _, _, done, _ = env.step(Action.UP)
                    if done:
                        env.reset()
                elif event.key == pygame.K_RIGHT:
                    _, _, done, _ = env.step(Action.RIGHT)
                    if done:
                        env.reset()
                elif event.key == pygame.K_DOWN:
                    _, _, done, _ = env.step(Action.DOWN)
                    if done:
                        env.reset()
                elif event.key == pygame.K_LEFT:
                    _, _, done, _ = env.step(Action.LEFT)
                    if done:
                        env.reset()

        # Render environment
        env.render()

    # Clean up
    env.close()


if __name__ == "__main__":
    main()
