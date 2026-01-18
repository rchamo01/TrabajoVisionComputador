import numpy as np
import tensorflow as tf
import os
import cv2
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tqdm import tqdm

# Modelos
from red_neuronal_vision import cargar_modelo_base as load_effnet
from red_neuronal_resnet import cargar_modelo_resnet as load_resnet

# Limpieza
tf.keras.backend.clear_session()

# Configuración
DATASET_ROOT = 'C:/Users/aborb/.cache/kagglehub/datasets/arnabkumarroy02/ferplus/versions/3'
VAL_DIR = os.path.join(DATASET_ROOT, 'validation')
TEST_DIR = os.path.join(DATASET_ROOT, 'test')

# Pequeño fix por si acaso
if not os.path.exists(VAL_DIR): VAL_DIR = os.path.join(DATASET_ROOT, 'validation')

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 8

# Cargar modelos preentrenados
print("Cargando EfficientNet...")
base_eff = load_effnet()
x = base_eff.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out_eff = Dense(NUM_CLASSES, activation='softmax')(x)
model_eff = Model(base_eff.input, out_eff)
model_eff.load_weights('mejor_modelo_ferplus_81.h5')

print("Cargando ResNet...")
base_res = load_resnet()
x = base_res.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x) 
x = Dropout(0.4)(x)
out_res = Dense(NUM_CLASSES, activation='softmax')(x)
model_res = Model(base_res.input, out_res)
model_res.load_weights('mejor_modelo_resnet.h5')

#  Funcion TTA 
def get_predictions_with_tta(model, directory, is_resnet=False):
    """
    Recorre un directorio, aplica TTA a cada imagen y devuelve las predicciones promediadas.
    """
    print(f"procesando {directory} con TTA (ResNet={is_resnet})...")
    
    
    generator = ImageDataGenerator().flow_from_directory(
        directory, target_size=(IMG_SIZE, IMG_SIZE), batch_size=1, 
        class_mode='sparse', shuffle=False
    )
    
    y_true = generator.classes
    filenames = generator.filenames
    final_preds = []

   
    for i in tqdm(range(len(filenames))):
        
        file_path = os.path.join(directory, filenames[i])
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Convertir 
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32) / 255.0
        
        
        v1 = img_tensor
        v2 = tf.image.flip_left_right(img_tensor)
        v3 = tf.image.adjust_brightness(img_tensor, 0.1)
        v4 = tf.image.adjust_contrast(img_tensor, 1.2)
        
        v5 = tf.image.central_crop(img_tensor, 0.9)
        v5 = tf.image.resize(v5, (IMG_SIZE, IMG_SIZE))
        
        batch_tta = tf.stack([v1, v2, v3, v4, v5]) #
        
        if is_resnet:
            batch_res = batch_tta*255.0
            batch_input = batch_res.numpy()
        else:

            batch_input = (batch_tta*255).numpy()

        #predicción
        preds_5 = model.predict(batch_input, verbose=0)
        pred_avg = np.mean(preds_5, axis=0) 
        
        final_preds.append(pred_avg)

    return np.array(final_preds), y_true



# Obtener datos de validación
print("\n Datos de validacion TTA")
val_preds_eff, y_val = get_predictions_with_tta(model_eff, VAL_DIR, is_resnet=False)
val_preds_res, _     = get_predictions_with_tta(model_res, VAL_DIR, is_resnet=True)

# Obtener datos de test
print("\n Datos de test TTA")
test_preds_eff, y_test = get_predictions_with_tta(model_eff, TEST_DIR, is_resnet=False)
test_preds_res, _      = get_predictions_with_tta(model_res, TEST_DIR, is_resnet=True)

# Optimización

# Estrategia A
print("\n Mezcla ponderada")
best_acc = 0
best_w = 0
for w in np.arange(0, 1.01, 0.05):
    mezcla = (w * val_preds_eff) + ((1 - w) * val_preds_res)
    acc = accuracy_score(y_val, np.argmax(mezcla, axis=1))
    if acc > best_acc:
        best_acc = acc
        best_w = w

# Aplicamos al test
mezcla_test = (best_w * test_preds_eff) + ((1 - best_w) * test_preds_res)
acc_test_weighted = accuracy_score(y_test, np.argmax(mezcla_test, axis=1))

# Estrategia B
print("\n Tercera red")
X_train_stack = np.concatenate([val_preds_eff, val_preds_res], axis=1)
X_test_stack = np.concatenate([test_preds_eff, test_preds_res], axis=1)

juez = LogisticRegression(max_iter=1000)
juez.fit(X_train_stack, y_val)

preds_stack = juez.predict(X_test_stack)
acc_test_stack = accuracy_score(y_test, preds_stack)

# --- Resultados ---

print(f"1. EfficientNet + TTA:      {accuracy_score(y_test, np.argmax(test_preds_eff, axis=1))*100:.2f}%")
print(f"2. ResNet + TTA:            {accuracy_score(y_test, np.argmax(test_preds_res, axis=1))*100:.2f}%")