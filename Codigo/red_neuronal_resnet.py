import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
import os

def cargar_modelo_resnet():
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    IMG_SIZE = 224

    print("Cargando modelo base ResNet50V2...")

    # 1. Definimos la entrada
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # 2. Preprocesamiento específico de ResNet
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)

    # 3. Cargamos la ResNet conectada a esa entrada
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_tensor=x 
    )

    # 4. Congelamos
    base_model.trainable = False
    
    return base_model