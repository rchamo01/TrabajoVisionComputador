import numpy as np
import tensorflow as tf

# Configuración de la GPU 

print("Configurando GPU...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU configurada: {len(gpus)} encontrada(s)")
    except RuntimeError as e:
        print(f"Error GPU: {e}")
else:
    print("❌ No hay GPU, usaremos CPU.")

import os
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import preprocess_input
from red_neuronal_resnet import cargar_modelo_resnet 

# Párametros 
IMG_SIZE = 224
BATCH_SIZE = 8 
DATASET_ROOT = 'C:/Users/aborb/.cache/kagglehub/datasets/arnabkumarroy02/ferplus/versions/3'
train_dir = os.path.join(DATASET_ROOT, 'train')
test_dir = os.path.join(DATASET_ROOT, 'test')

#   Generadores
train_datagen = ImageDataGenerator(
    rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.1, horizontal_flip=True
)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb',
    batch_size=BATCH_SIZE, class_mode='sparse'
)
validation_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb',
    batch_size=BATCH_SIZE, class_mode='sparse', shuffle=False
)
NUM_CLASSES = train_generator.num_classes

# Modelos Resnet
base_model = cargar_modelo_resnet()
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x) 
x = Dropout(0.4)(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# Callbacks personalizados
class SanitizeLogs(tf.keras.callbacks.Callback):
    """
    Para limpiar los callbacks
    """
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


# Callback para guardar pesos
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
            # Guardamos solo los pesos
            
            self.model.save_weights(self.filepath, overwrite=True)



# Instanciamos nuestro guardador seguro
checkpoint_seguro = SafeModelCheckpoint(
    filepath='mejor_modelo_resnet.h5',
    monitor='val_accuracy',
    verbose=1
)

# 2. Reducir velocidad si nos estancamos
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,      # Bajar Lr
    patience=3,      # Cada 3 épocas sin mejora
    min_lr=1e-7,
    verbose=1
)

# 3. Parar si  se ve observa sobreajuste
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)
sanitize_callback = SanitizeLogs()
callbacks_list = [sanitize_callback,checkpoint_seguro, reduce_lr, early_stop]

callbacks = [sanitize_callback, checkpoint_seguro, reduce_lr, early_stop]
fase= 0
epocas_fase1 = 5
if fase == 1:
    # FASE 1
    print("\n--- FASE 1: RESNET CONGELADA ---")
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=epocas_fase1, validation_data=validation_generator, callbacks=callbacks)
    # Guardamos pesos intermedios
    model.save_weights('modelo_resnet_fase1.h5')

else:

    model.load_weights('mejor_modelo_resnet.h5')


base_model.trainable = True
# Fase 2
print("\n--- FASE 2: RESNET FINE-TUNING ---")
fine_tune_at = 140

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

trainable_count = sum([1 for layer in base_model.layers if layer.trainable])


model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5), 
    loss='sparse_categorical_crossentropy', metrics=['accuracy']
)
model.fit(train_generator, epochs=30, validation_data=validation_generator, 
          callbacks=callbacks, initial_epoch=epocas_fase1)

model.save_weights('modelo_resnet_FINAL.h5')
print("¡Entrenamiento terminado!")