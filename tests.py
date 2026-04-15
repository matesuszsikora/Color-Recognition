
import numpy as np
# Test Lab conversion
rgb_color = np.array([0.5, 0.2, 0.8]) # Example RGB color
lab_color = convert_lab(rgb_color)
print(f"RGB: {rgb_color}")
print(f"Lab: {lab_color}")

# Convert back to RGB to check
rgb_back = lab_to_rgb(lab_color)
print(f"RGB back from Lab: {rgb_back}")

# Test normalization and unnormalization
normalized_lab = lab_normalize(lab_color)
print(f"Normalized Lab: {normalized_lab}")

unnormalized_lab = lab_unnorm(normalized_lab)
print(f"Unnormalized Lab: {unnormalized_lab}")

# Test with hex
hex_color = "ff0000" # Red
rgb_from_hex = hex_to_rgb(hex_color)
print(f"Hex: {hex_color}, RGB from hex: {rgb_from_hex}")

# Test hex to rgb tuple
rgb_tuple_from_hex = hex_to_rgb_tuple("#"+hex_color)
print(f"Hex: {hex_color}, RGB tuple from hex: {rgb_tuple_from_hex}")

# Test rgb to hex
hex_from_rgb = rgb_to_hex(rgb_from_hex)
print(f"RGB: {rgb_from_hex}, Hex from RGB: {hex_from_rgb}")

