import numpy as np
import pandas as pd
import os 
import cv2
import random
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import warnings 
warnings.filterwarnings('ignore')
path='/kaggle/input/leapgestrecog/leapGestRecog/'
folders=os.listdir(path)
images = []
labels = []
for folder in folders:
    folder_path = os.path.join(path,folder)
    subfolders = os.listdir(folder_path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path,subfolder)
            
        for img in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path,img)
            images.append(img_path)
            labels.append(subfolder)
#display random images with their labels             
random_indices = random.sample(range(len(images)),20)
random_imges =  [images[i] for i in random_indices]
random_labels = [labels[i] for i in random_indices]
fig, axes = plt.subplots(5, 4, figsize=(30, 20 ))
axes = axes.flatten()
for idx,(img,label)  in enumerate(zip(random_imges,random_labels)):
    img=Image.open(img)
    img_array=np.array(img)
    axes[idx].imshow(img_array)
    axes[idx].set_title(label)
    axes[idx].axis('off')
    
plt.show()    
#creat a DataFrame    
df = pd.DataFrame({'images':images,'labels':labels})    
# mapping the labels
le = LabelEncoder()
df['labels'] = le.fit_transform(df['labels'])
x = []
y = np.array(df['labels'])
y=to_categorical(y)
# resize the images
img_size = (64, 64)
for i,img in enumerate(df['images']):
    img = Image.open(img)
    img = img.convert('RGB')
    img_resized = img.resize(img_size)
    img_array = np.array(img_resized)
    img_normalized = img_array /255.0
    x.append(img_normalized)
x = np.array(x)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=54)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=3, 
                               restore_best_weights=True, 
                               verbose=1) 

history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                    validation_data=(x_test, y_test), 
                    callbacks=[early_stopping])


print("Summary Of The Model")
model.summary()
print("Predictions")
predictions = model.predict(x_test)
predicted_classes =  np.argmax(predictions , axis =1)
test_loss,test_accuracy = model.evaluate(x_test,y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

print("*MODEL SUMMARY*")
model.summary()


print("TEST RESULTS")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss     : {test_loss}")
print(f"Test Accuracy : {test_accuracy}")


print("MODEL PREDICTIONS")
predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

print(f"Number of predicted classes : {len(np.unique(predicted_classes))}")
print(f"Number of actual classes    : {len(np.unique(true_classes))}")


print("CLASSIFICATION REPORT")
print(classification_report(true_classes, predicted_classes))
random_indices = random.sample(range(len(x_test)), 10)

fig, axes = plt.subplots(5, 2, figsize=(8, 16))  
axes = axes.flatten()

for i, idx in enumerate(random_indices):
    img = x_test[idx]
    actual_class = np.argmax(y_test[idx])
    predicted_class = predicted_classes[idx]

    actual_label = le.inverse_transform([actual_class])[0]
    predicted_label = le.inverse_transform([predicted_class])[0]

    axes[i].imshow(img)
    axes[i].set_title(f"Predicted: {predicted_label}\nActual: {actual_label}", fontsize=12)
    axes[i].axis('off')

plt.tight_layout()
plt.show()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

y_true = np.argmax(y_test, axis=1)
y_pred = predicted_classes

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_true, y_pred, target_names=le.classes_))