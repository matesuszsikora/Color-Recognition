# --- DATA PREPARATION ---
# Load and process your dataframe
input_colors = 5
basedir = "Data/PhotosColorPicker/"

df = make_dataset()
df['image_path'] = basedir + df['image']
def load_image(path):
    img = load_img(path, target_size=(128, 128))
    rgb_norm = img_to_array(img) / 255.0
    lab = convert_lab(rgb_norm)
    return lab_normalize(lab)

b = df.groupby(df['image']).sum()
b = b[['color_1','color_2','color_3','color_4','color_5']]
b = b.reset_index()
b['image_path'] = basedir + b['image']
from sklearn.cluster import KMeans

for c in range(1,6):
    b[f"norm_rgb_{c}"] = b[f'color_{c}'].apply(lambda a : [hex_to_rgb(str(i)) for i in a])
for c in range(1,6):
    b[f"lab_{c}"] = b[f'norm_rgb_{c}'].apply(lambda a : [convert_lab(np.array(i)) for i in a])

for c in range(1,6):
    b[f"norm_lab_{c}"] = b[f'lab_{c}'].apply(lambda a : [lab_normalize(np.array(i)) for i in a])

def cluster_image_colors(color_vectors, n_clusters=5):
    # Convert to numpy array
    color_array = np.array(color_vectors)

    # Handle edge cases: if fewer vectors than clusters
    if len(color_array) < n_clusters:
        return color_array  # Just return raw vectors

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(color_array)

    return kmeans.cluster_centers_

# Apply to each row (each image)
b['color_palette_lab'] = b[f'norm_lab_{input_colors}'].apply(lambda x: cluster_image_colors(x, n_clusters=input_colors))
b["color_paletter_rgb"] = b['color_palette_lab'].apply(
    lambda palette: [lab_to_rgb(lab_unnorm(color)) for color in palette]
)
c = b["color_paletter_rgb"]

image_tensors = tf.stack([load_image(path) for path in b['image_path']])
labs = np.stack(b['color_palette_lab'].values)
labels = np.array(labs.astype(np.float64)).reshape(-1, 3*input_colors)
# TF dataset
dataset = tf.data.Dataset.from_tensor_slices((image_tensors, labels)).shuffle(buffer_size=len(image_tensors))
val_size = int(0.2 * len(image_tensors))
train_dataset = dataset.skip(val_size).batch(32)
val_dataset = dataset.take(val_size).batch(32)
