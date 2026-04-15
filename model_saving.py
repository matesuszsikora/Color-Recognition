model.fit(train_dataset, epochs=10, validation_data=val_dataset)

model.save("model_5v3_colors.keras")

print(history)