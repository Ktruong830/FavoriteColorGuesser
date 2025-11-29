import colorsys
import numpy as np
import math

# Converts RGB values to HSV values
# R[0-255], G[0-255], B[0-255] --> H[0-360), S[0-100], V[0-100]
def rgb_to_hsv(rgb_values):
    r, g, b = [val / 255.0 for val in rgb_values] # Normalizes RGB values
    h, s, v = colorsys.rgb_to_hsv(r, g, b) # Gets normalized HSV values
    hsv_values = [                         # Converts back to ordinary HSV values
        max(0, min(int(round(h * 360)), 359)),
        max(0, min(int(round(s * 100)), 100)),
        max(0, min(int(round(v * 100)), 100))
    ]
    return hsv_values

# Converts HSV values to RGB values
# H[0-360), S[0-100], V[0-100] --> R[0-255], G[0-255], B[0-255]
def hsv_to_rgb(hsv_values):
    h, s, v = hsv_values
    r, g, b = colorsys.hsv_to_rgb(h / 360, s / 100, v / 100) # Gets normalized RGB values
    rgb_values = [                                           # Converts back to ordinary RGB values
        max(0, min(int(round(r * 255)), 255)),
        max(0, min(int(round(g * 255)), 255)),
        max(0, min(int(round(b * 255)), 255))
    ]
    return rgb_values

# Converts HSV values to circular HSV values
def hsv_to_circular(hsv_values):
    h, s, v = hsv_values
    h = math.radians(h) # Converts h to radians
    circular_values = [math.sin(h), math.cos(h), s, v] # Gets circular values with the hue represented by sin and cos
    return circular_values

# Converts circular HSV values to HSV values
def circular_to_hsv(circular_values):
    sin_h, cos_h, s, v = circular_values
    h = math.atan2(sin_h, cos_h) # Converts the sin and cos representation of h to radians
    if h < 0: # Ensures h is represented as positive
        h += 2 * math.pi
    h = round(math.degrees(h)) % 360 # Converts h to degrees
    hsv_values = [h, s, v]
    return hsv_values

# Denormalizes model output and returns the color in either RBG or HSV
def denormalize_model_output(pred_circ, color_model):
    predicted_circular = np.array(pred_circ.copy())
    # Reverts normalization (sin/cos hue in [-1,1], saturation/value in [0,100])
    predicted_circular[0] = np.clip(predicted_circular[0], -1, 1)
    predicted_circular[1] = np.clip(predicted_circular[1], -1, 1)
    predicted_circular[2] = np.clip(predicted_circular[2] * 100, 0,
                                    100)  # Converts saturation back to percentage and clamp
    predicted_circular[3] = np.clip(predicted_circular[3] * 100, 0,
                                    100)  # Converts value back to percentage and clamp
    # Converts the circular HSV to regular HSV
    predicted_hsv = circular_to_hsv(predicted_circular)
    predicted_hsv[0] = int(round(predicted_hsv[0])) % 360
    predicted_hsv[1] = int(np.clip(round(predicted_hsv[1]), 0, 100))
    predicted_hsv[2] = int(np.clip(round(predicted_hsv[2]), 0, 100))
    if color_model == "HSV": # Returns if HSV is selected model
        return predicted_hsv
    elif color_model == "RGB": # Returns if RGB is selected model
        predicted_rgb = hsv_to_rgb(predicted_hsv)
        predicted_rgb[0] = int(np.clip(round(predicted_rgb[0]), 0, 255))
        predicted_rgb[1] = int(np.clip(round(predicted_rgb[1]), 0, 255))
        predicted_rgb[2] = int(np.clip(round(predicted_rgb[2]), 0, 255))
        return predicted_rgb
    else:
        raise ValueError("Invalid color model chosen. Must be 'RGB' or 'HSV'.")