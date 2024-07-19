"""
Auteur : AURELIEN PINON - PINA20100100
Fichier contenant le modèle CNN (Convolutives Neural Networks)
"""
import tensorflow as tf

from Model import Model

from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy, CategoricalAccuracy, MeanAbsoluteError, RootMeanSquaredError
from keras.losses import MeanSquaredError, BinaryCrossentropy, CategoricalCrossentropy


class CNN(Model):
    """
    Modèle de prédiction d'âge, de genre et d'ethnie reposant sur le principe de CNN
    """
    def __init__(self, input_shape=(200, 200, 3), age_max=110):
        """
        Constructeur du CNN
        :param input_shape: Taille des images d'entrées par défaut (200*200*3)
        :param age_max: Age max que le CNN devra reconnaître
        """
        super().__init__(input_shape, age_max)

    def build_model(self):
        """
        Construction du modèle CNN
        :return: Le modèle crée, compilé et prêt à être entraîné
        """
        # Récupération de l'entrée
        entree = Input(shape=self.input_shape)

        # 1ère couche de convolution
        age = Conv2D(16, (3, 3), padding="same")(entree)
        age = BatchNormalization()(age)
        age = Activation('relu')(age)
        age = MaxPooling2D((2, 2))(age)
        age = Dropout(0.5)(age)

        genre = Conv2D(16, (3, 3), padding="same")(entree)
        genre = BatchNormalization()(genre)
        genre = Activation('relu')(genre)
        genre = MaxPooling2D((2, 2))(genre)
        genre = Dropout(0.5)(genre)

        ethnie = Conv2D(16, (3, 3), padding="same")(entree)
        ethnie = BatchNormalization()(ethnie)
        ethnie = Activation('relu')(ethnie)
        ethnie = MaxPooling2D((2, 2))(ethnie)
        ethnie = Dropout(0.5)(ethnie)

        # 2ème couche de convolution
        age = Conv2D(32, (3, 3), padding="same")(age)
        age = BatchNormalization()(age)
        age = Activation('relu')(age)
        age = MaxPooling2D((2, 2))(age)
        age = Dropout(0.5)(age)

        genre = Conv2D(32, (3, 3), padding="same")(genre)
        genre = BatchNormalization()(genre)
        genre = Activation('relu')(genre)
        genre = MaxPooling2D((2, 2))(genre)
        genre = Dropout(0.5)(genre)

        ethnie = Conv2D(32, (3, 3), padding="same")(ethnie)
        ethnie = BatchNormalization()(ethnie)
        ethnie = Activation('relu')(ethnie)
        ethnie = MaxPooling2D((2, 2))(ethnie)
        ethnie = Dropout(0.5)(ethnie)

        # 3ème couche de convolution
        age = Conv2D(64, (3, 3), padding="same")(age)
        age = BatchNormalization()(age)
        age = Activation('relu')(age)
        age = MaxPooling2D((2, 2))(age)
        age = Dropout(0.5)(age)

        genre = Conv2D(64, (3, 3), padding="same")(genre)
        genre = BatchNormalization()(genre)
        genre = Activation('relu')(genre)
        genre = MaxPooling2D((2, 2))(genre)
        genre = Dropout(0.5)(genre)

        ethnie = Conv2D(64, (3, 3), padding="same")(ethnie)
        ethnie = BatchNormalization()(ethnie)
        ethnie = Activation('relu')(ethnie)
        ethnie = MaxPooling2D((2, 2))(ethnie)
        ethnie = Dropout(0.5)(ethnie)

        # Applatissement des données en un vecteur
        age = Flatten()(age)
        genre = Flatten()(genre)
        ethnie = Flatten()(ethnie)

        # 1 ère couche Dense
        age = Dense(256)(age)
        age = BatchNormalization()(age)
        age = Activation('relu')(age)
        dropout_age = Dropout(0.5)(age)

        genre = Dense(256)(genre)
        genre = BatchNormalization()(genre)
        genre = Activation('relu')(genre)
        dropout_genre = Dropout(0.5)(genre)

        ethnie = Dense(256)(ethnie)
        ethnie = BatchNormalization()(ethnie)
        ethnie = Activation('relu')(ethnie)
        dropout_ethnie = Dropout(0.5)(ethnie)

        # Prédiction
        age = layers.Dense(1, activation='linear', name='age')(dropout_age)
        genre = layers.Dense(2, activation='sigmoid', name='genre')(dropout_genre)
        ethnie = layers.Dense(5, activation='softmax', name='ethnie')(dropout_ethnie)

        # Construction et compilation du modèle
        model = tf.keras.Model(inputs=entree, outputs=[age, genre, ethnie])
        optimizer = Adam(learning_rate=0.0005)
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