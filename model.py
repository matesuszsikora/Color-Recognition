

model = models.Sequential([
     layers.Input(shape=(128, 128, 3)),
     layers.Conv2D(16 * input_colors, (3, 3), activation='relu'),
     layers.MaxPooling2D(2, 2),

     layers.Conv2D(32 * input_colors, (3, 3), activation='relu'),
     layers.MaxPooling2D(2, 2),

     layers.Conv2D(64 * input_colors, (3, 3), activation='relu'),
     layers.Flatten(),
     layers.Dropout(0.5),
    layers.Dense(3 * input_colors, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=1e-3), loss='mse')
history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)


