import numpy as np

# Realsense D455
CAMERA_CONFIG = {
    "rgb":{
        "width": 1280,
        "height": 800,
        "fovy": 65, #deg
        "fovx": 90, # deg

    },
    "depth":{
        "width": 1280,
        "height": 720,
        "fovy": 58, #deg
        "fovx": 87, # deg
    }
}