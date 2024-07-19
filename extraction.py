"""
Auteur : AURELIEN PINON - PINA20100100
Fichier contenant les fonctions relatives à la création des dataframes et à l'extraction des données.
"""
import os
import numpy as np
import cv2
import pandas as pd
import constante as CS

from alive_progress import alive_bar
from dataprep import compter_nb_image
from keras.utils import to_categorical

def get_dataframe(NOM_DOSSIER, type='Train', normalize=False):
    """
    Créer le dataframe relatif à un dossier (TRAIN ou TEST ou VALIDATION)
    :param NOM_DOSSIER: Nom du chemin pour accéder au dossier contenant les images à traiter
    :return: Un dataframe contenant toutes les informations relatives à une image.
                index = nom_fichier
                image = matrice contenant les pixels relatifs à l'image
                age = Entier correspondant à l'âge de la personne sur l'image
                genre = Entier (0 ou 1) 0 = Homme, 1 = Femme
                ethnie = Entier (0 à 4) 0 = Caucasien, 1 = Africain, 2 = Asiatique, 3 = Indien, 4 = Autre
    """
    # Création des tableaux qui contiendront toutes les informations sur les images
    index  = []
    image  = []
    age    = []
    genre  = []
    ethnie = []

    # Parcours des images contenues dans le dossier passé en paramètre
    with alive_bar(total=compter_nb_image(NOM_DOSSIER), title="Création du dataframe " + type,
                   theme='smooth') as bar:
        for fichier in os.listdir(NOM_DOSSIER):
            # Affichage de l'avancement
            bar.text = f"Working on Image : {fichier} !"
            bar()

            # Ouverture de l'image

            img = cv2.imread(NOM_DOSSIER + '/' + fichier)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Extractions des infos
            year, sexe, ethnicity, etat = get_info(fichier)

            # Copie des infos dans les tableaux
            if etat == CS.SUCCES:
                index.append(fichier)
                genre.append(sexe)
                ethnie.append(ethnicity)
                if normalize:
                    image.append(img/255.0)
                    age.append(year/CS.AGE_MAX)
                else:
                    image.append(img)
                    age.append(year)

    dataframe = pd.DataFrame({'Image': image, 'Age': age, 'Genre': genre, 'Ethnie': ethnie}, index=index)
    dataframe_melange = dataframe.sample(frac=1)

    return dataframe_melange

def get_info(img_name):
    """
    A partir du nom d'une image, extrait les infos nécessaires
    :param img_name: nom de l'image
    :return: Age = Entier, Sexe = Entier (0 ou 1), Ethnie = Entier (0 à 4), SUCCES ou ECHEC de l'extraction
    """
    info = img_name.split('_')
    if len(info) < 4:
        return -1, -1, -1, CS.ECHEC

    age = int(info[0])
    sexe = int(info[1])
    ethnie = int(info[2])

    return age, sexe, ethnie, CS.SUCCES


def get_ethnie(ethnie):
    """
    Transforme la colonne du tableau contenant l'ethnie de la personne sur l'image en sa représentation textuelle
    :param ethnie: Colonne Ethnie d'un dataframe
    :return: La colonne du dataframe en représentation textuelle
    """
    for index, elem in enumerate(ethnie):
        if elem == 0:   ethnie[index] = 'Caucasien'
        elif elem == 1: ethnie[index] = 'Africain'
        elif elem == 2: ethnie[index] = 'Asiatique'
        elif elem == 3: ethnie[index] = 'Indien'
        else:           ethnie[index] = 'Autre'

    return ethnie

def get_genre(genre):
    """
    Transforme la colonne du tableau contenant le genre de la personne sur l'image en sa représentation textuelle
    :param genre: Colonne Genre d'un dataframe
    :return: La colonne du dataframe en représentation textuelle
    """
    for index, elem in enumerate(genre):
        if elem == 0: genre[index] = 'Homme'
        else:         genre[index] = 'Femme'

    return genre

def generer_entrer(df, entrainement=True, batch_size=16):
    """
    Générateur utilisé lors de l'entraînement des modèles
    :param df: Un dataframe contenant les variables de validation ou d'entraînement
    :param entrainement: Booléen permettant de mettre fin à la génération
    :param batch_size: Taille de Batch utilisé pour l'entraînement
    :return:
    """
    # Tableau contenant les données d'un batch
    images, ages, genres, ethnies = [], [], [], []
    while True:
        for index, row in df.iterrows():
            # Preprocessing des données
            img, age, genre, ethnie = preprocess(row['Image'], row['Age'], row['Genre'], row['Ethnie'])
            # Copie des données dans les tableau
            images.append(img)
            ages.append(age)
            genres.append(genre)
            ethnies.append(ethnie)
            # Fournit le batch d'entraînement au modèle
            if len(images) >= batch_size:
                yield np.array(images), [np.array(ages), np.array(genres), np.array(ethnies)]
                # Remise à 0 des tableaux
                images, ages, genres, ethnies = [], [], [], []

        if not entrainement:
            break

def preprocess(img, age, genre, ethnie):
    """
    Effectue le preprocessing des données. Normalise les pixels de l'image, Transforme l'âge en float compris
    entre 0 et 1 et catégorise le genre et l'ethnie selon le nombre de classe possible (2 pour le genre, 5
    pour l'ethnie).
    :param img: Tableaux contenant les pixels d'une image
    :param age: Entier compris entre 1 et AGE_MAX
    :param genre: 0 = Homme, 1 = Femme
    :param ethnie: Entier (0 à 4) 0 = Caucasien, 1 = Africain, 2 = Asiatique, 3 = Indien, 4 = Autre
    :return: Les données avec le preprocessing effectué dans l'ordre suivant img, age, genre, ethnie
    """
    img = np.array(img) / 255.0
    age = (age-1)/(CS.AGE_MAX-1)
    genre = to_categorical(genre, 2)
    ethnie = to_categorical(ethnie, 5)

    return img, age, genre, ethnie
