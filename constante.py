"""
Auteur : AURELIEN PINON - PINA20100100
Fichier contenant les constantes relatives au projet
"""
import os
import warnings

from keras.callbacks import ModelCheckpoint

warnings.filterwarnings("ignore")

def clear():
    os.system('cls' if os.name=='nt' else 'clear')


# Nom de Dossier
DOSSIER_IMAGES_PRETRAITEES = 'Images/Images_pretraitees_v5'
DOSSIER_IMAGES_BRUTS       = 'Images/Images_brut'
DOSSIER_IMAGES_PREPAREES   = 'Images/Images_preparees'
DOSSIER_TRAIN = "Images/Train"
DOSSIER_TEST = "Images/Test"
DOSSIER_VALIDATION = "Images/Validation"

# Détecteur et prédicteurs
FRONTAL_FACE_DETECTOR = "Detecteurs/haarcascade_frontalface_default.xml"
EYE_DETECTOR          = "Detecteurs/haarcascade_eye.xml"
PREDICTOR             = "Detecteurs/shape_predictor_5_face_landmarks.dat"

# Variables de traitement
NOUVEAU_FORMAT = (200, 200)
SENS_HORAIRE = -1
ANTI_HORAIRE = 1

# Variables d'état
SUCCES = 'Succès'
ECHEC  = 'Echec'
ALL = 'all'
AGE = 'age'
GENRE = 'genre'
ETHNIE = 'ethnie'

# Variables d'entraînement
AGE_MAX = 110
BATCH_SIZE = 32
EPOCHS = 25

# Variables relatives aux modèles
NUMERO_MODEL_CNN = 10
NUMERO_MODEL_VGG = 3
NUMERO_MODEL_XCEPTION = 2

# Callbacks de sauvegarde
CHECKPOINT_PATH_CNN = "CNN/Weights/Model" + str(NUMERO_MODEL_CNN) + "/cp-{epoch:02d}.h5"
CHECKPOINT_CALLBACK_CNN = ModelCheckpoint(
    filepath=CHECKPOINT_PATH_CNN,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    save_freq='epoch')

CHECKPOINT_PATH_VGG = "VGG/Weights/Model" + str(NUMERO_MODEL_VGG) + "/cp-{epoch:02d}.h5"
CHECKPOINT_CALLBACK_VGG = ModelCheckpoint(
    filepath=CHECKPOINT_PATH_VGG,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    save_freq='epoch')

CHECKPOINT_PATH_XCEPTION = "XCEPTION/Weights/Model" + str(NUMERO_MODEL_XCEPTION) + "/cp-{epoch:02d}.h5"
CHECKPOINT_CALLBACK_XCEPTION = ModelCheckpoint(
    filepath=CHECKPOINT_PATH_XCEPTION,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    save_freq='epoch')