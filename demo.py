from model import HairToolModel
from generate_stl import create_comb_stl

# Hair profiles
new_user = [[3,2,2,0,1,1]]

# Predicting dimensions
spacing = 8.3
length = 44.1
angle = 29
roundness = 0.95

# Generating STL
create_comb_stl(
    spacing,
    length,
    angle,
    roundness
)
