from __future__ import annotations

import numpy as np
import random 
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Ball, WorldObj
# from minigrid.envs import Empty

from minigrid.core.constants import COLOR_NAMES 
from minigrid.minigrid_env import MiniGridEnv
from minigrid.manual_control import ManualControl

class CustomEmpty(WorldObj):
    def __init__(self, color: str, states=None):
        print(f"CustomEmpty: Initializing with color={color}")
        super().__init__("empty", color)
        self.contains = None
        self.can_overlap = lambda: True
        print(f"CustomEmpty: Finished init for color={color}")



class RoomBall(Ball):
    def __init__(self, color: str, states=None):
        print(f"RoomBall: Initializing with color={color}") 
        super().__init__(color)
        self.can_overlap = lambda: True
        print(f"RoomBall: Finished init for color={color}") 


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

        
        mission_space = MissionSpace(mission_func=lambda: "Exploration")

        
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
        print("--- _gen_grid: Started ---") 
        self.grid = Grid(width, height)
        print(f"--- _gen_grid: Grid initialized: {width}x{height} ---") 
        self.grid.wall_rect(0, 0, width, height)
        print("--- _gen_grid: Walls added ---") 
        WORLD_OBJECT_TYPES = [CustomEmpty, RoomBall]
        NOVEL_COLORS = [c for c in COLOR_NAMES if c not in ['black', 'white', 'grey']]
        if not NOVEL_COLORS:
            NOVEL_COLORS = [c for c in COLOR_NAMES if c != 'black']
        print(f"--- _gen_grid: Novel colors selected: {NOVEL_COLORS} ---") 
        for i in range(width):
            for j in range(height):
                print(f"--- _gen_grid: Checking cell ({i}, {j}) ---") 
                if self.grid.get(i, j) is None:
                    obj_type = random.choice(WORLD_OBJECT_TYPES)
                    obj_color = random.choice(NOVEL_COLORS)
                    print(f"--- _gen_grid: Attempting to place object: Type={obj_type.__name__}, Color={obj_color} at ({i}, {j}) ---") 
                    self.grid.set(i, j, obj_type(obj_color))
                    print(f"--- _gen_grid: Object placed: Type={obj_type.__name__}, Color={obj_color} at ({i}, {j}) ---") 
                else:
                    print(f"--- _gen_grid: Cell ({i}, {j}) already occupied ---") 
        print("--- _gen_grid: All objects placement loop finished ---") 

        print("--- _gen_grid: Calling place_agent() ---") 
        self.place_agent()
        print("--- _gen_grid: Agent placed ---") 
        self.mission = "Exploration"
        print("--- _gen_grid: Mission set ---")
        print("--- _gen_grid: Finished ---") 



# Independent environment testing
if __name__ == "__main__":
    print("Testing GridEnv directly...")
    try:
        env = GridEnv(render_mode="human")
        env.reset()
        manual_control = ManualControl(env, seed=42)
        manual_control.start()
    except Exception as e:
        print(f"An error occurred during environment setup or manual control: {e}")
        import traceback
        traceback.print_exc() 
    print("GridEnv test finished.")