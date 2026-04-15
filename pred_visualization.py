# --- PREDICTION VISUALIZATION ---

# Predict on new image
#model = keras.saving.load_model("model.keras")
def show_colors(hex_list, titles=None):
    fig, ax = plt.subplots(1, len(hex_list), figsize=(len(hex_list) * 2, 2))
    if len(hex_list) == 1:
        ax = [ax]
    for i, hex_color in enumerate(hex_list):
        rgb = np.array([[hex_to_rgb_tuple(hex_color)]])
        ax[i].imshow(rgb)
        ax[i].axis("off")
        if titles:
            ax[i].set_title(titles[i])
    plt.tight_layout()
    plt.show()
file = "000000461404.jpg"
image_path = f"Data/PhotosColorPicker/{file}"
x
image = load_img(image_path, target_size=(128, 128))
img_array = img_to_array(image) / 255.0
lab_image = convert_lab(img_array)
lab_normed = lab_normalize(lab_image)
input_arr = np.expand_dims(lab_normed, axis=0)
prediction = model.predict(input_arr).reshape(input_colors,3)
lab_pred = [lab_unnorm(i) for i in prediction]
rgb_pred = [lab_to_rgb(i) for i in lab_pred]
pred_hex = [rgb_to_hex(i) for i in rgb_pred]

true_rgb = list(b.loc[file, "color_paletter_rgb"])
#true_rgb = [i for i in b[b['image'] == file]["color_paletter_rgb"].iloc[0]]
true_hex = [rgb_to_hex(i) for i in true_rgb]
print("Predicted color:", pred_hex)
print("True color:", true_hex)
titles = [f"Predicted {i+1}" for i in range(len(pred_hex))]
show_colors(pred_hex, titles)
titles = [f"True {i+1}" for i in range(len(true_hex))]
show_colors(true_hex, titles)s