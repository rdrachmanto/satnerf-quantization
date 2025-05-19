# import satellite_cloud_generator as scg
import sys

sys.path.append("/data/rdr78068/satnerf-base/SatelliteCloudGenerator")

from src import *

def inject_cloud(img):
    print("Injecting CLOUD")
    img, _, _ = add_cloud_and_shadow(img, return_cloud=True)
    return img


def inject_thick_fog(img):
    img, _ = add_cloud(img, min_lvl=0.5, max_lvl=0.9, decay_factor=1.85, return_cloud=True)
    return img


def inject_thin_fog(img):
    img, _ = add_cloud(img,
                           min_lvl=0.0,
                           max_lvl=0.5,
                           cloud_color=False,
                           channel_offset=0,
                           blur_scaling=4.0,
                           return_cloud=True)
    return img
