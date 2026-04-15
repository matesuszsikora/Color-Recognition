a = make_dataset()
b = a.groupby(a['image']).sum()
b = b[['color_1','color_2','color_3','color_4','color_5']]

from sklearn.cluster import KMeans

clusters = 5

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
b['color_palette_lab'] = b[f'norm_lab_{clusters}'].apply(lambda x: cluster_image_colors(x, n_clusters=clusters))
b["color_paletter_rgb"] = b['color_palette_lab'].apply(
    lambda palette: [lab_to_rgb(lab_unnorm(color)) for color in palette]
)
c = b["color_paletter_rgb"]
print(b['color_palette_lab'])
