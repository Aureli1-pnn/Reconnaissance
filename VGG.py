"""
Auteur : AURELIEN PINON - PINA20100100
Fichier contenant le modèle se basant sur VGG
"""
import tensorflow as tf

from Model import Model

from keras.applications.vgg16 import VGG16
from keras import layers
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy, CategoricalAccuracy, MeanAbsoluteError, RootMeanSquaredError
from keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy


class AgeGenreEtEthniePredictionUtilisantVGG(Model):
    """
    Modèle de prédiction d'âge, de genre et d'ethnie reposant sur le modèle pré-entraîné VGG16
    """
    def __init__(self, input_shape=(200, 200, 3), age_max=110):
        """
        Constructeur du CNN
        :param input_shape: Taille des images d'entrées par défaut (200*200*3)
        :param age_max: Age max que le modèle devra reconnaître
        """
        super().__init__(input_shape, age_max)

    def build_model(self):
        """
        Construction du modèle
        :return: Le modèle crée et prêt à être entraîné
        """
        model_vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=self.input_shape)
        model_vgg16.trainable = False
        features = model_vgg16.output

        # Applatissement des données en un vecteur
        features = Flatten()(features)

        # 1 ère couche Dense

        age = Dense(256)(features)
        age = BatchNormalization()(age)
        age = Activation('relu')(age)
        dropout_age = Dropout(0.5)(age)

        genre = Dense(256)(features)
        genre = BatchNormalization()(genre)
        genre = Activation('relu')(genre)
        dropout_genre = Dropout(0.5)(genre)

        ethnie = Dense(256)(features)
        ethnie = BatchNormalization()(ethnie)
        ethnie = Activation('relu')(ethnie)
        dropout_ethnie = Dropout(0.5)(ethnie)

        # Prédiction
        age = layers.Dense(1, activation='linear', name='age')(dropout_age)
        genre = layers.Dense(2, activation='sigmoid', name='genre')(dropout_genre)
        ethnie = layers.Dense(5, activation='softmax', name='ethnie')(dropout_ethnie)

        # Construction et compilation du modèle
        model = tf.keras.Model(inputs=model_vgg16.inputs, outputs=[age, genre, ethnie])
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer,
                      loss={
                            'age': MeanSquaredError(),
                            'genre': BinaryCrossentropy(),
                            'ethnie': CategoricalCrossentropy()
                      },
                      loss_weights={
                            'age': 4,
                            'genre': 2,
                            'ethnie': 1
                      },
                      metrics={ 'age': [MeanAbsoluteError(), RootMeanSquaredError()],
                                'genre': BinaryAccuracy(threshold=0.5),
                                'ethnie': CategoricalAccuracy()
                      })

        return model