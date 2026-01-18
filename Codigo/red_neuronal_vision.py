
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
import os


def cargar_modelo_base():
   
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    
    IMG_SIZE = 224 #numero de pixeles para imagen

    # Cargar modelo

    print("Cargando modelo base EfficientNetB0...")

    try:
        base_model = EfficientNetB0(
            include_top=False,          
            weights='imagenet',         
            input_shape=(IMG_SIZE, IMG_SIZE, 3) 
        )

        #congelar capas
        base_model.trainable = False

        return base_model

    except Exception as e:
        print(f"\nHa ocurrido un error al cargar el modelo: {e}")
        print("Por favor, asegúrate de tener conexión a internet la primera vez que lo ejecutas.")
        print("Verifica también que tu versión de TensorFlow es compatible.")



