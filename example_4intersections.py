import numpy as np
from world import World
from agents import Car, RectangleBuilding, Painting
from geometry import Point

# World settings
dt = 0.1  # time steps in seconds
world_width = 120  # meters
world_height = 120
lane_width = 3.5
sidewalk_width = 2.0

# Create world
w = World(dt, width=world_width, height=world_height, ppm=6)

# Street dimensions
street_width = 2 * lane_width  # Two lanes per direction
block_size = 40  # Size of building blocks

# Create horizontal and vertical roads (as paintings)
# Horizontal roads
for y in [world_height/4, 3*world_height/4]:
    w.add(Painting(
        Point(world_width/2, y),
        Point(world_width, 2*street_width),
        'gray20'
    ))

# Vertical roads
for x in [world_width/4, 3*world_width/4]:
    w.add(Painting(
        Point(x, world_height/2),
        Point(2*street_width, world_height),
        'gray20'
    ))

# Add lane markers
marker_width = 0.5
marker_length = 5
gap_length = 5
marker_color = 'white'

# Helper function to add dashed lines
def add_dashed_line(start_x, start_y, length, is_vertical=False):
    current_pos = 0
    while current_pos < length:
        marker_pos = min(marker_length, length - current_pos)
        if is_vertical:
            w.add(Painting(
                Point(start_x, start_y + current_pos + marker_pos/2),
                Point(marker_width, marker_pos),
                marker_color
            ))
        else:
            w.add(Painting(
                Point(start_x + current_pos + marker_pos/2, start_y),
                Point(marker_pos, marker_width),
                marker_color
            ))
        current_pos += marker_pos + gap_length

# Add lane markers for horizontal roads
for y in [world_height/4, 3*world_height/4]:
    add_dashed_line(0, y, world_width)

# Add lane markers for vertical roads
for x in [world_width/4, 3*world_width/4]:
    add_dashed_line(x, 0, world_height, is_vertical=True)

# Add buildings in the corners and middle blocks
building_positions = [
    # Corner buildings
    (block_size/2, block_size/2),
    (block_size/2, world_height - block_size/2),
    (world_width - block_size/2, block_size/2),
    (world_width - block_size/2, world_height - block_size/2),
    # Middle buildings
    (block_size/2, world_height/2),
    (world_width - block_size/2, world_height/2),
    (world_width/2, block_size/2),
    (world_width/2, world_height - block_size/2),
    (world_width/2, world_height/2)
]

for pos_x, pos_y in building_positions:
    building = RectangleBuilding(
        Point(pos_x, pos_y),
        Point(block_size - sidewalk_width, block_size - sidewalk_width),
        'gray80'
    )
    w.add(building)

# Add a car
c1 = Car(Point(world_width/4 - lane_width/2, 10), np.pi/2)
c1.max_speed = 30.0  # 30 m/s (108 km/h)
c1.velocity = Point(0, 3.0)
w.add(c1)

if __name__ == "__main__":
    # Simple visualization loop
    from interactive_controllers import KeyboardController
    
    c1.set_control(0., 0.)
    controller = KeyboardController(w)
    
    try:
        for _ in range(600):
            c1.set_control(controller.steering, controller.throttle)
            w.tick()
            w.render()
            if w.collision_exists():
                break
    finally:
        w.close() 