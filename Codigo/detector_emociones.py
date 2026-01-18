import os
import cv2 
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from red_neuronal_vision import cargar_modelo_base

# GPU
print("TensorFlow:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Intenta configurar TensorFlow para que no reserve toda la memoria
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Se encontraron {len(gpus)} GPUs:")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")
    except RuntimeError as e:
        print(f"Error al configurar la GPU: {e}")
else:
    print("No se detectan GPUs.")
    


# Callbacks

class SanitizeLogs(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            for key, value in logs.items():
                if tf.is_tensor(value):
                    logs[key] = float(value)
        
        if hasattr(self.model, 'history') and hasattr(self.model.history, 'history'):
            history_dict = self.model.history.history
            for key, values in history_dict.items():
                new_values = []
                for v in values:
                    if tf.is_tensor(v):
                        new_values.append(float(v))
                    else:
                        new_values.append(v)
                history_dict[key] = new_values


class SafeModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_accuracy', verbose=1):
        super(SafeModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.best = -np.inf  

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if current > self.best:
            if self.verbose > 0:
                print(f'\nEpoch {epoch+1}: {self.monitor} mejoró de {self.best:.5f} a {current:.5f}, guardando modelo en {self.filepath}')
            self.best = current
            # Guardar solo los pesos del modelo
            self.model.save_weights(self.filepath, overwrite=True)

# Instancia callback guardar
checkpoint_seguro = SafeModelCheckpoint(
    filepath='mejor_modelo_ferplus.h5',
    monitor='val_accuracy',
    verbose=1
)

# reductor veolocidad
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,      
    patience=3,      
    min_lr=1e-7,
    verbose=1
)

# para parar 
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)
sanitize_callback = SanitizeLogs()
callbacks_list = [sanitize_callback,checkpoint_seguro, reduce_lr, early_stop]
# Parametros
IMG_SIZE = 224  
BATCH_SIZE = 8 


#DATASET_ROOT = 'C:/Users/aborb/.cache/kagglehub/datasets/ananthu017/emotion-detection-fer/versions/1/' 
DATASET_ROOT = 'C:/Users/aborb/.cache/kagglehub/datasets/arnabkumarroy02/ferplus/versions/3'
train_dir = os.path.join(DATASET_ROOT, 'train')
test_dir = os.path.join(DATASET_ROOT, 'test')

# Generador
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(        # Normaliza píxeles de 0-255 a 0-1
    rotation_range=10,     # Aumento de datos: gira un poco
    width_shift_range=0.1, # Aumento de datos: desplaza
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

#generador de prueba
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

# Conectar generadores

print("Cargando Train")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),    
    color_mode='rgb',                    
    batch_size=BATCH_SIZE,
    class_mode='sparse'                  
)

print("\nCargando Validacion")
validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False                        
)

# Que clases tenemos
print(f"\nClases encontradas: {list(train_generator.class_indices.keys())}")
NUM_CLASSES = train_generator.num_classes

# Cargar Modelo

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

fase1=True
if fase1:
    modelo=cargar_modelo_base()
    
    x = modelo.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x) 

    
    model = Model(inputs=modelo.input, outputs=outputs)

    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    print("\nModelo completo y compilado:")
    model.summary()

    #Entrenamiento del modelo

    import matplotlib.pyplot as plt


    EPOCHS = 10 # Empecemos con 10 épocas

    print("\n Inicio de Entrenamiento")

    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    print("\nFin del Entrenamiento")

    

    # Gráfico de Precisión 
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión (train)')
    plt.plot(history.history['val_accuracy'], label='Precisión (test/val)')
    plt.title('Gráfico de Precisión')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()


    # Gráfico de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida (train)')
    plt.plot(history.history['val_loss'], label='Pérdida (test/val)')
    plt.title('Gráfico de Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()


    # Guardado gráficos
    plt.savefig('grafico_fase1.png')
    plt.close() 
    print("Gráfico de Fase 1 guardado en 'grafico_fase1.png'")

    # Guardado modelo
    model.save_weights('modelo_emociones_v1_weights.h5')
    print("Modelo guardado en 'modelo_emociones_v1.h5'")

    # Guardamos el diccionario 
    import pickle
    history_fase1_data = history.history
    with open('history_fase1.pkl', 'wb') as f:
        pickle.dump(history_fase1_data, f)

    

print("\nEntrenamiento fase 2")


import pickle
with open('history_fase1.pkl', 'rb') as f:
    history_fase1_data = pickle.load(f)

for key, values in history_fase1_data.items():
    
    cleaned_values = []
    for v in values:
        
        if hasattr(v, 'numpy'): 
            cleaned_values.append(float(v.numpy()))
        elif isinstance(v, (np.ndarray, np.generic)):
            cleaned_values.append(float(v))
        else:
            cleaned_values.append(float(v))
    
    history_fase1_data[key] = cleaned_values

# Reconstruccion
modelo_base = cargar_modelo_base() 
x = modelo_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=modelo_base.input, outputs=outputs)

# Cargando pesos
print("Cargando pesos de la Fase 1")

model.load_weights('modelo_emociones_v1_weights.h5') 

# Descongelamos
modelo_base.trainable = True 

# Compilamos de nuevo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Modelo re-compilado para Fine-Tuning (base descongelada).")
model.summary() 

# Mas epocas para el fine tuning

EPOCHS_EXTRA = 25
initial_epoch = len(history_fase1_data['accuracy'])
total_epochs = initial_epoch + EPOCHS_EXTRA

history_fine_tuning = model.fit(
    train_generator,
    epochs=total_epochs,
    validation_data=validation_generator,
    callbacks=callbacks_list,
    initial_epoch=initial_epoch 
)

# Guardado final
model.save_weights('modelo_emociones_FINAL.h5')
print("¡Modelo FINAL guardado en 'modelo_emociones_FINAL.h5'!")



if 'accuracy' in history_fase1_data:
    acc_key = 'accuracy'
    val_acc_key = 'val_accuracy'
    print("Clave detectada: 'accuracy'")
else:
    acc_key = 'acc'
    val_acc_key = 'val_acc'
    print("Clave detectada: 'acc'")

# Usamos la clave para las listas
acc = history_fase1_data[acc_key] + history_fine_tuning.history[acc_key]
val_acc = history_fase1_data[val_acc_key] + history_fine_tuning.history[val_acc_key]

loss = history_fase1_data['loss'] + history_fine_tuning.history['loss']
val_loss = history_fase1_data['val_loss'] + history_fine_tuning.history['val_loss']

# Grafica total
initial_epochs = len(history_fase1_data[acc_key]) 

plt.figure(figsize=(16, 6))

# --- Gráfico de Precisión 
plt.subplot(1, 2, 1)
plt.plot(acc, label='Precisión (Entrenamiento)')
plt.plot(val_acc, label='Precisión (Validación)')
plt.axvline(initial_epochs - 1, color='red', linestyle='--', label='Inicio Fine-Tuning')
plt.title('Gráfico de Precisión Total (Fase 1 + Fine-Tuning)')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

# --- Gráfico de Pérdida 
plt.subplot(1, 2, 2)
plt.plot(loss, label='Pérdida (Entrenamiento)')
plt.plot(val_loss, label='Pérdida (Validación)')
plt.axvline(initial_epochs - 1, color='red', linestyle='--', label='Inicio Fine-Tuning')
plt.title('Gráfico de Pérdida Total (Fase 1 + Fine-Tuning)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.show()
plt.savefig('grafico_total.png')