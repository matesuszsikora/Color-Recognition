import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.utils import load_img, img_to_array
from matplotlib import pyplot as plt
import os
import re
import pandas as pd
from collections import defaultdict
import matplotlib.colors as mcolors

# --- COLOR SPACE UTILITIES ---
def hex_to_rgb(hex_color):
    return [int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4)]

def f(t):
    delta = 6/29
    return np.where(t > delta**3, np.cbrt(t), (t / (3 * delta**2)) + (4/29))

def inv_f(t):
    delta = 6/29
    return np.where(t > delta, t**3, 3 * delta**2 * (t - 4/29))

def inv_gamma_correct(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1/2.4) - 0.055)

def lab_normalize(lab):
    return (lab + np.array([0, 128, 128])) / np.array([100, 255, 255])

def lab_unnorm(lab):
    return lab * np.array([100, 255, 255]) - np.array([0, 128, 128])

def lab_to_rgb(lab):
    L, a, b = lab
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    xyz = np.array([
        0.95047 * inv_f(fx),
        1.00000 * inv_f(fy),
        1.08883 * inv_f(fz)
    ])
    rgb_lin = np.array([
        3.2406 * xyz[0] - 1.5372 * xyz[1] - 0.4986 * xyz[2],
        -0.9689 * xyz[0] + 1.8758 * xyz[1] + 0.0415 * xyz[2],
        0.0557 * xyz[0] - 0.2040 * xyz[1] + 1.0570 * xyz[2]
    ])
    rgb = inv_gamma_correct(np.clip(rgb_lin, 0, 1))
    return np.clip(rgb, 0, 1)

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def convert_lab(image):
    mask = image > 0.04045
    img_linear = np.where(mask, ((image + 0.055) / 1.055) ** 2.4, image / 12.92)
    R, G, B = img_linear[..., 0], img_linear[..., 1], img_linear[..., 2]
    X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.950489
    Y = (0.2126729 * R + 0.7151522 * G + 0.0721750 * B) / 1.0
    Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.088840
    X, Y, Z = f(X), f(Y), f(Z)
    L = 116.0 * Y - 16.0
    a = 500.0 * (X - Y)
    b = 200.0 * (Y - Z)
    return np.stack([L, a, b], axis=-1)

def make_dataset():
    basedir = "Data/Res_ColorPickerCustomPicker"
    rows = []

    for file in os.listdir(basedir):
        filepath = os.path.join(basedir, file)
        if re.match(r".*\d{2}\.txt", file):
            image_groups = defaultdict(list)  # key: image filename, value: list of color groups

            with open(filepath, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    image = parts[0]
                    colors = [c.strip(",").replace('#', '') for c in parts[1:]]
                    image_groups[image].append(colors)

            # Now for each image in this file, pad its color groups and add to the rows
            for image, color_groups in image_groups.items():
                while len(color_groups) < 5:
                    color_groups.append([])

                rows.append({
                    'file': file,
                    'image': image,
                    'color_1': color_groups[0],
                    'color_2': color_groups[1],
                    'color_3': color_groups[2],
                    'color_4': color_groups[3],
                    'color_5': color_groups[4],
                })

    return pd.DataFrame(rows)



def hex_to_rgb_tuple(hex_color):
    return mcolors.to_rgb(hex_color