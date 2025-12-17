

# 1. Prepare Data
print("🚀 Loading and Splitting Datasets...")
train_ds = prepare_dataset(f"{DATA_DIR}/train")
val_ds = prepare_dataset(f"{DATA_DIR}/val")
test_ds = prepare_dataset(f"{DATA_DIR}/test")

# 2. Build and Train Model
print("🏗️ Building Multi-Output CNN...")
model = build_split_cnn()

# Using EarlyStopping to prevent overfitting
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("🧠 Starting Training...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[callback])

# 3. Concatenate Predictions for Test Set
print("📊 Evaluating and Concatenating Predictions...")

# Get ground truth labels
y_true_combined = []
for _, labels in test_ds:
    # Re-combine three separate labels back into one integer
    combined = labels['out_1'] * 100 + labels['out_2'] * 10 + labels['out_3']
    y_true_combined.extend(combined.numpy())

# Get model predictions (returns a list of 3 arrays)
preds = model.predict(test_ds)
p1 = np.argmax(preds[0], axis=1)
p2 = np.argmax(preds[1], axis=1)
p3 = np.argmax(preds[2], axis=1)

y_pred_combined = p1 * 100 + p2 * 10 + p3

# 4. Final Metrics
final_acc = accuracy_score(y_true_combined, y_pred_combined)
print(f"\n✅ FINAL CONCATENATED TEST ACCURACY: {final_acc:.4f}")

# 5. Visualizations
plt.figure(figsize=(12, 5))

# Plot Training Accuracy (taking average of the three heads)
avg_val_acc = (np.array(history.history['val_out_1_accuracy']) +
               np.array(history.history['val_out_2_accuracy']) +
               np.array(history.history['val_out_3_accuracy'])) / 3

plt.subplot(1, 2, 1)
plt.plot(avg_val_acc, label='Average Val Accuracy')
plt.title('Multi-Output Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Confusion Matrix for the "Hundreds" digit (Head 1) as a sample
plt.subplot(1, 2, 2)
# Re-extract actual hundreds digit for the first head
y_true_h = [y // 100 for y in y_true_combined]
cm = confusion_matrix(y_true_h, p1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (First Digit Head)')

plt.tight_layout()
plt.show()