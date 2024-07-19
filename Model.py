"""
Auteur : AURELIEN PINON - PINA20100100
Fichier contenant la classe modèle qui sera la classe parent de tout nos modèles
"""
import tensorflow as tf
import abc
from keras.utils.vis_utils import plot_model

#from numba import jit

class Model(metaclass=abc.ABCMeta):
    """
    Class Model abstraite, elle sera le parent de chaque modèle
    """
    def __init__(self, input_shape, age_max):
        """
        Constructeur du CNN
        :param input_shape: Taille des images d'entrées par défaut (200*200*3)
        :param age_max: Age max que le modèle devra reconnaître
        """
        self.input_shape = input_shape
        self.age_max = age_max
        self.model = self.build_model()

    @abc.abstractmethod
    def build_model(self):
        pass

    #@jit
    def train(self, ds_train, nb_donnees_train, nb_donnees_valid,
              epochs=10, batch_size=16, verbose=1, callbacks=None, valid_data=None):
        """
        Entraînement du modèle
        :param train_images: Images d'entraînement
        :param age_labels: Ages correspondant aux images
        :param genre_labels: Genres correspondant aux images
        :param ethnie_labels: Ethnies correspondant aux images
        :param epochs: Nombre d'epochs lors du train
        :param batch_size: Taille de batch
        :return: L'évolution des données du modèle (loss et métriques)
        """

        return self.model.fit(ds_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              steps_per_epoch=nb_donnees_train // batch_size,
                              callbacks=callbacks,
                              validation_data=valid_data,
                              validation_steps=nb_donnees_valid // batch_size,
                              verbose=verbose)

    def evaluate(self, ds_test, verbose=1, batch_size=32, taille_test=None):
        """
        Evaluation du modèle avec les images de test
        :param ds_test: Générateur fournissant X et Y à chaque batch
        :param verbose: Gère l'affichage lors de l'entraînement
        :param batch_size: Taille de batch
        :param taille_test: Nombre de données de test
        :return: L'évaluation du modèle avec les métriques stipulaient lors de la compilation du modèle
        """
        return self.model.evaluate(ds_test, verbose=verbose, batch_size=batch_size, steps=taille_test//batch_size)

    def predict(self, images):
        """
        Effectue la prédiction pour les images passées en paramètre
        :param images: Vecteurs d'images
        :return: Une prédiction du modèle pour chaque image
        """
        return self.model.predict(images, verbose=0)

    def decode_prediction(self, prediction):
        """
        Retourne l'âge estimé ainsi que le genre et l'ethnie avec le pourcentage de confiance de ces deux derniers
        :param predictions: tableau contenant les predictions du modèle
        :return: Age -> entier
                 Genre -> [Homme ou Femme, % de confiance]
                 Ethnie -> [Caucasien ou Africain ou ..., % de confiance]
        """

        age = int(round((prediction[0][0][0] * self.age_max) + 1, 0))
        if age < 0: age = abs(age)
        genres = prediction[1][0]
        ethnies = prediction[2][0]

        if genres[0] > genres[1]: genre = 'Homme'; genre_confiance = round(genres[0]*100, 2)
        else: genre = 'Femme'; genre_confiance = round(genres[1]*100, 2)

        label = 0
        ethnie_confiance = -1
        ethnie = 0
        for confiance in ethnies:
            if confiance > ethnie_confiance: ethnie_confiance = confiance; ethnie = label
            label += 1
        ethnie_confiance = round(ethnie_confiance*100, 2)

        if ethnie == 0: ethnie = 'Caucasien'
        elif ethnie == 1: ethnie = 'Africain'
        elif ethnie == 2: ethnie = 'Asiatique'
        elif ethnie == 3: ethnie = 'Indien'
        else: ethnie = 'Autre'

        return age, [genre, genre_confiance], [ethnie, ethnie_confiance]

    def summary(self):
        """
        Affiche le sommaire du modèle
        :return: None
        """
        print(self.model.summary())

    def save_model(self, PATH):
        """
        Sauvegarde le modèle dans le chemin passé en argument
        :param PATH: Dossier où le modèle sera enregistré
        :return: None
        """
        self.model.save(PATH)

    def load_model(self, PATH):
        """
        Charge un modèle pré-entrainé contenu dans PATH et l'enregistre dans self.model
        :param PATH: chemin menant au modèle
        :return: None
        """
        self.model = tf.keras.models.load_model(PATH)

    def load_weights(self, PATH):
        """
        Charge les poids contenu dans PATH dans self.model
        :param PATH: chemin menant au fichier contenant les poids
        :return: None
        """
        self.model.load_weights(PATH)

    def plot_model(self, to_file='model_plot.png', show_shapes=True, show_layer_names=True):
        """
        Créer une image représentant le modèle
        :param to_file: Nom du fichier
        :param show_shapes: Booléen, affiche les tailles des différents layers
        :param show_layer_names: Booléen, affiche les noms des différents layers
        :return: None
        """
        plot_model(self.model, to_file=to_file, show_shapes=show_shapes, show_layer_names=show_layer_names)