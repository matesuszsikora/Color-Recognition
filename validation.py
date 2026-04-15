
import matplotlib.pyplot as plt

# Zmieniamy rozmiar wykresu
plt.figure(figsize=(12, 5))

# Wykres straty (loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
min_val_loss = min(history.history['val_loss'])
min_epoch = history.history['val_loss'].index(min_val_loss)
plt.axvline(x=min_epoch, linestyle='--', color='r', label=f'Min Val Loss (epoch {min_epoch})')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Wykres dokładności (accuracy), jeśli dostępna
if 'accuracy' in history.history:
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    max_val_acc = max(history.history['val_accuracy'])
    max_epoch = history.history['val_accuracy'].index(max_val_acc)
    plt.axvline(x=max_epoch, linestyle='--', color='g', label=f'Max Val Acc (epoch {max_epoch})')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()