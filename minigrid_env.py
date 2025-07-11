from __future__ import annotations

import numpy as np 
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall 
from minigrid.minigrid_env import MiniGridEnv
from minigrid.manual_control import ManualControl


class GridEnv(MiniGridEnv):
    def __init__(
        self,
        size=8, # 8 x 8 grid
        agent_start_pos=None, # Agent placed randomly
        agent_start_dir=3, # Agent right facing
        max_steps: int | None = None,
        agent_view_size=9,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        
        mission_space = MissionSpace(mission_func=lambda: "")

        # Max steps
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False, 
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.place_agent()
        self.mission = "Explore the grid"



# Independent environment test

if __name__ == "__main__":
    print("Testing BareBonesGridEnv directly...")
    env = GridEnv(render_mode="human")
    manual_control = ManualControl(env, seed=42)
    manual_control.start()
    print("Environment test finished.")