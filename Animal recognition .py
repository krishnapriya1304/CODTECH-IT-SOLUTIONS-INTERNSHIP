#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

train_dir = r'C:\Users\nivi1\Downloads\archive (2)\animals\animallll'
val_dir = r'C:\Users\nivi1\Downloads\archive (2)\animals\animallll'

# Loading the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False)

# Adding custom layers for fine-tuning
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  

model = Model(inputs=base_model.input, outputs=predictions)

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Preparing the data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Training the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=10
)

# Save the trained model
model.save('model_animal.keras')


# In[8]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model_animal.keras")

results = {0: 'bear', 1:'cat',2:'cow',3:'deer',4:'dog',5:'elephant',6:'goat',7:'lion',8:'tiger',9:'zebra'}  

def predict_image(image_path):
    im = Image.open(image_path)
    im = im.resize((32, 32))
    im = np.expand_dims(im, axis=0)
    im = np.array(im)
    im = im / 255.0 
    pred = np.argmax(model.predict(im), axis=-1)[0]
    return results.get(pred, "Unknown")

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((500, 500))
        img = ImageTk.PhotoImage(img)
        
        image_label.configure(image=img)
        image_label.image = img
        

        predicted_animal = predict_image(file_path)
        result_label.config(text=f"Predicted Animal: {predicted_animal}")

# Create the main window
window = tk.Tk()
window.title("Animal Classifier")

window.attributes("-fullscreen", True)

def exit_fullscreen(event=None):
    window.attributes("-fullscreen", False)
    return "break"

window.bind("<Escape>", exit_fullscreen)

# Button to upload image
upload_button = tk.Button(window, text="Upload Image", font=("Arial", 26), command=open_image)
upload_button.pack(pady=80)

# Label to display the image
image_label = tk.Label(window)
image_label.pack(pady=20)

# Label to display the result
result_label = tk.Label(window, text="Predicted Animal: ", font=("Arial", 26))
result_label.pack(pady=20)

window.mainloop()


# In[13]:





# In[ ]:





# In[ ]:




