# John Modl, iEdge 2025 REU
# General code flow and structure based on Vice et al. (https://github.com/jackvice/lstm_explore)

# Environment created with help from Google Gemini. 



from __future__ import annotations
from gymnasium.envs.registration import register
import numpy as np
import random
# Import Box from minigrid.core.world_object
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall, Ball, WorldObj, Door, Box
from minigrid.core.constants import COLOR_NAMES
from minigrid.minigrid_env import MiniGridEnv
from minigrid.manual_control import ManualControl

class CustomEmpty(WorldObj):
    def __init__(self, color: str, states = None):
        super().__init__("empty",color)
        self.contains = None
        self.can_overlap = lambda: False
    


class RoomBall(Ball):
    def __init__(self, color: str, states=None):
        super().__init__(color)
        self.can_overlap = lambda: False 


class RoomSquare(Box): 
    def __init__(self, color: str, states=None):
        super().__init__(color) 
        self.can_overlap = lambda: False 
    


class GridEnv(MiniGridEnv):
    def __init__(
        self,
        size=27, # 25 X 25
        agent_start_pos=None,
        agent_start_dir=3,
        enable_intrinsic_reward=False,
        **kwargs,
    ):
        self.enable_intrinsic_reward = enable_intrinsic_reward

        super().__init__(
            mission_space = MissionSpace(mission_func=lambda: "Explore all rooms"),
            grid_size= size,
            max_steps=4*size*size,
            **kwargs
        )

        self.visit_counts = np.zeros((self.width, self.height), dtype=int)
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

    def _reset(self, seed=None, options=None):
        obs, info = super()._reset(seed=seed, options=options)
        self.visit_counts = np.zeros((self.width, self.height), dtype=int)
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.visit_counts[self.agent_pos[0], self.agent_pos[1]] += 1
        return obs, reward, terminated, truncated, info

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height) # Outer walls

        
        
        hallway_width = 3
        room_depth = (height - 2 - hallway_width) // 2
        room_width = (width - 2 - hallway_width) // 2

        # Central Hallway (vertical)
        hallway_x_start = width // 2 - hallway_width // 2
        hallway_x_end = hallway_x_start + hallway_width -1

        # Left side rooms (connected)
        room1_x_start, room1_y_start = 1, 1
        room1_x_end, room1_y_end = hallway_x_start - 1, room_depth

        room2_x_start, room2_y_start = 1, room_depth + hallway_width + 1
        room2_x_end, room2_y_end = hallway_x_start - 1, height - 2

        # Right side rooms (separate entrances)
        room3_x_start, room3_y_start = hallway_x_end + 1, 1
        room3_x_end, room3_y_end = width - 2, room_depth

        room4_x_start, room4_y_start = hallway_x_end + 1, room_depth + hallway_width + 1
        room4_x_end, room4_y_end = width - 2, height - 2

        # Draw walls for rooms
        self.grid.wall_rect(room1_x_start, room1_y_start, room1_x_end - room1_x_start + 1, room1_y_end - room1_y_start + 1)
        self.grid.wall_rect(room2_x_start, room2_y_start, room2_x_end - room2_x_start + 1, room2_y_end - room2_y_start + 1)
        self.grid.wall_rect(room3_x_start, room3_y_start, room3_x_end - room3_x_start + 1, room3_y_end - room3_y_start + 1)
        self.grid.wall_rect(room4_x_start, room4_y_start, room4_x_end - room4_x_start + 1, room4_y_end - room4_y_start + 1)

        # Create central hallway
        for y in range(1, height - 1):
            for x in range(hallway_x_start, hallway_x_end + 1):
                self.grid.set(x, y, None) # Clear cells for hallway

        # Connect rooms to hallway
        # Room 3 to hallway (direct)
        self.grid.set(hallway_x_end + 1, room3_y_start + room_depth // 2, None) # Opening on left wall of room 3
        self.grid.set(hallway_x_end, room3_y_start + room_depth // 2, None) # Corresponding opening in hallway

        # Room 4 to hallway (direct)
        self.grid.set(hallway_x_end + 1, room4_y_start + room_depth // 2, None) # Opening on left wall of room 4
        self.grid.set(hallway_x_end, room4_y_start + room_depth // 2, None) # Corresponding opening in hallway


        # Connect Room 1 and Room 2 on the left side, with a single entrance/exit to central hallway
        # 2. Create internal connection between Room1 and Room2
        internal_passage_x = (room1_x_start + room1_x_end) // 2
        for y in range(room1_y_end + 1, room2_y_start):
             self.grid.set(internal_passage_x, y, None)
        self.grid.set(internal_passage_x, room1_y_end, None) # Door from Room1
        self.grid.set(internal_passage_x, room2_y_start, None) # Door to Room2

        # 3. Create a single entrance/exit for the left connected pair to the central hallway
        single_left_entry_y = room2_y_start + room_depth // 2
        self.grid.set(hallway_x_start - 1, single_left_entry_y, None) # Opening in the wall of Room2
        self.grid.set(hallway_x_start, single_left_entry_y, None) # Corresponding opening in the hallway wall


        # Place colored circles and squares randomly in each of the rooms
        # Helper function to place objects in a given rectangular region
        def place_objects_in_room(x_min, y_min, x_max, y_max, num_objects=5):
            object_types = [RoomBall, RoomSquare]
            novel_colors = [c for c in COLOR_NAMES if c not in ['black', 'white', 'grey']]

            valid_positions = []
            for x in range(x_min + 1, x_max):
                for y in range(y_min + 1, y_max):
                    if self.grid.get(x, y) is None: # Cell is empty
                        valid_positions.append((x, y))

            if not valid_positions:
                return # No place to put objects
            
            num_to_place = min(num_objects, len(valid_positions)) 
            placements = random.sample(valid_positions, num_to_place)
            
            # Ensure there are valid coordinates to place objects
            # if x_min >= x_max or y_min >= y_max:
                # return

            for pos in placements:
                obj_type = random.choice(object_types)
                obj_color = random.choice(novel_colors)
                self.grid.set(pos[0], pos[1], obj_type(obj_color))


        # Place objects in each defined room area
        place_objects_in_room(room1_x_start, room1_y_start, room1_x_end, room1_y_end, num_objects=5)
        place_objects_in_room(room2_x_start, room2_y_start, room2_x_end, room2_y_end, num_objects=5)
        place_objects_in_room(room3_x_start, room3_y_start, room3_x_end, room3_y_end, num_objects=5)
        place_objects_in_room(room4_x_start, room4_y_start, room4_x_end, room4_y_end, num_objects=5)


        # Place agent randomly within the central hallway
        # Find all empty cells in the hallway
        hallway_cells = []
        for x in range(hallway_x_start, hallway_x_end + 1):
            for y in range(1, height - 1): # Exclude outer walls
                if self.grid.get(x, y) is None:
                    hallway_cells.append((x, y))

        if not hallway_cells:
            raise Exception("No empty cells in hallway for agent placement!")

        self.agent_start_pos = random.choice(hallway_cells)
        self.place_agent()
        self.mission = "Explore all rooms"

        # Calculate the true number of navigable cells after generation
        self.num_navigable_cells = 0
        for x in range(width):
            for y in range(height):
                cell = self.grid.get(x, y)
                # A cell is navigable if it's empty.
                if cell is None:
                    self.num_navigable_cells += 1
                



register(
    id='MiniGrid-Custom-Grid-v0',
    entry_point='minigrid_env:GridEnv',
)


# Independent environment testing
if __name__ == "__main__":
    print("Testing GridEnv directly with custom layout...")
    try:
        env = GridEnv(render_mode="human", size=27) # Ensure size matches expected grid
        env.reset()
        manual_control = ManualControl(env, seed=42)
        manual_control.start()
    except Exception as e:
        print(f"Environment setup or manual control error: {e}")
        import traceback
        traceback.print_exc()
    print("GridEnv test finished.")