"""
@author: Delia Martínez
"""
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# --- IMPORTACIONES DE MODELOS ---
from red_neuronal_vision import cargar_modelo_base as load_effnet
from red_neuronal_resnet import cargar_modelo_resnet as load_resnet

# Limpieza de sesión
tf.keras.backend.clear_session()

# --- CONFIGURACIÓN ---
DATASET_TEST_DIR = 'C:/Users/usuario/Documents/Master_UPM/Vision_por_computador/Entrega_final/test'
IMG_SIZE = 224
CLASS_NAMES = sorted(os.listdir(DATASET_TEST_DIR)) 
NUM_CLASSES = len(CLASS_NAMES)

print(f"📂 Clases detectadas: {CLASS_NAMES}")

# --- 1. CARGAR EL "DREAM TEAM" ---
print("\n--- CARGANDO MODELOS ---")

# A) EfficientNet
base_eff = load_effnet()
x = base_eff.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out_eff = Dense(NUM_CLASSES, activation='softmax')(x)
model_eff = Model(base_eff.input, out_eff)
try:
    model_eff.load_weights('mejor_modelo_ferplus_81.h5')
    print("✅ EfficientNet cargada.")
except:
    print("❌ ERROR: Faltan pesos EfficientNet")

# B) ResNet
base_res = load_resnet()
x = base_res.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x) 
x = Dropout(0.4)(x)
out_res = Dense(NUM_CLASSES, activation='softmax')(x)
model_res = Model(base_res.input, out_res)
try:
    model_res.load_weights('mejor_modelo_resnet.h5')
    print("✅ ResNet cargada.")
except:
    print("❌ ERROR: Faltan pesos ResNet")


# --- 2. SELECCIÓN DE IMÁGENES ALEATORIAS ( 10 ejemplos) ---
print("\n--- SELECCIONANDO 10 IMÁGENES AL AZAR ---")

all_test_images = []
for label in CLASS_NAMES:
    folder = os.path.join(DATASET_TEST_DIR, label)
    if os.path.exists(folder):
        for f in os.listdir(folder):
            full_path = os.path.join(folder, f)
            all_test_images.append((full_path, label))

muestras = random.sample(all_test_images, 10)


# --- 3. BUCLE SIMPLE (Visualización de 10 ejemplos) ---
plt.figure(figsize=(20, 10)) 

for i, (img_path, true_label) in enumerate(muestras):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_tensor = np.expand_dims(img_resized, axis=0)
    
    preds_eff = model_eff.predict(img_tensor, verbose=0)
    preds_res = model_res.predict(img_tensor, verbose=0)
    final_pred = (preds_eff + preds_res) / 2
    
    pred_idx = np.argmax(final_pred)
    pred_label = CLASS_NAMES[pred_idx]
    confidence = final_pred[0][pred_idx] * 100 
    
    plt.subplot(2, 5, i + 1)
    plt.imshow(img_resized)
    plt.axis('off')
    
    color_titulo = 'green' if true_label == pred_label else 'red'
    titulo = f"Real: {true_label}\nIA: {pred_label}\n({confidence:.1f}%)"
    plt.title(titulo, color=color_titulo, fontsize=12, fontweight='bold')

print("\n📸 Generando galería de resultados...")
plt.tight_layout()
plt.show()

# ==========================================================
# --- 4. FASE DE VALIDACIÓN COMPLETA ---
# ==========================================================
print("\n" + "="*50)
print("📊 INICIANDO VALIDACIÓN SOBRE EL DATASET COMPLETO")
print("="*50)

y_true_all = []
y_pred_all = []

# Procesamos TODAS las imágenes guardadas en test
for i, (img_path, true_label) in enumerate(all_test_images):
    # Feedback del progreso cada 100 imágenes
    if i % 100 == 0:
        print(f"Procesando imagen {i}/{len(all_test_images)}...")
    
    img = cv2.imread(img_path)
    if img is None: continue
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_tensor = np.expand_dims(img_resized, axis=0)
    
    # Inferencia Ensamble
    preds_eff = model_eff.predict(img_tensor, verbose=0)
    preds_res = model_res.predict(img_tensor, verbose=0)
    final_pred = (preds_eff + preds_res) / 2
    
    # Guardar resultados para métricas
    y_true_all.append(CLASS_NAMES.index(true_label))
    y_pred_all.append(np.argmax(final_pred))

# --- A) INFORME DE MÉTRICAS ---
print("\n📝 INFORME DE CLASIFICACIÓN POR CLASE:")
print(classification_report(y_true_all, y_pred_all, target_names=CLASS_NAMES))

# --- B) MATRIZ DE CONFUSIÓN ---
cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Matriz de Confusión - Ensamble Final', fontsize=16)
plt.ylabel('Etiqueta Real (Ground Truth)', fontsize=12)
plt.xlabel('Predicción de la Red Neuronal', fontsize=12)
plt.tight_layout()
plt.savefig('Confusion_Matrix_Final.png') # Se guarda en la carpeta del script
plt.show()

print("\n✅ Validación terminada. Se ha guardado 'Confusion_Matrix_Final.png' con los resultados.")

