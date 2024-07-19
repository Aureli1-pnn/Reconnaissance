"""
Auteur : AURELIEN PINON - PINA20100100
Fichier contenant les fonctions relatives à la visualisation des données.
"""
import constante as CS
import matplotlib.pyplot as plt

from extraction import get_dataframe
from extraction import get_genre
from extraction import get_ethnie

def afficher_info_age(df_train, df_test, df_valid):
    """
    Affiche les infos sous forme d'histogramme et de camembert
    :param df_train: dataframe contenant les informations du dossier train
    :param df_test: dataframe contenant les informations du dossier test
    :param df_valid: dataframe contenant les informations du dossier validation
    :return: None
    """
    # Création des subplots
    plt.plot(figsize=(15, 8))

    # Affichage des âges en un histogramme
    plt.hist([df_train.Age, df_test.Age, df_valid.Age],
             bins=[0,10,20,30,40,50,60,70,80,90,100,110],
             color = ['red', 'blue', 'green'],
             edgecolor='white',
             label=['Train', 'Test', 'Validation'])
    plt.xlabel('Age par tranche de 10 ans')
    plt.ylabel("Nombres d'observations")
    plt.legend()
    plt.title("Âge")
    plt.show()

def afficher_info_genre_et_ethnie(df, nom_dossier='Train'):
    """
    Affiche les infos sous forme d'histogramme et de camembert
    :param df: dataframe contenant les informations d'un des dossiers train, test ou validation
    :param nom_dossier: Nom du dossier traité (Train, test ou validation)
    :return: None
    """

    # Création des subplots
    fig, ax_un = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
    ax1, ax2 = ax_un

    # Transformation des caractéristiques sous forme d'entier en text
    genre  = get_genre(df.Genre)
    ethnie = get_ethnie(df.Ethnie)

    # Affichage de la répartition des Genre
    proportion_genre = genre.value_counts()
    labels_genre = proportion_genre.index
    ax1.pie(proportion_genre, labels = labels_genre, startangle=90, shadow=True, autopct='%1.1f%%')
    ax1.axis('equal')
    ax1.set_title("Genre")

    # Affichage de la répartition des Ethnies
    proportion_ethnie = ethnie.value_counts()
    labels_ethnie = proportion_ethnie.index
    ax2.pie(proportion_ethnie, labels=labels_ethnie, startangle=90, shadow=True, autopct='%1.1f%%')
    ax2.axis('equal')
    ax2.set_title("Ethnie")

    plt.show()

def afficher_historique_d_entrainement(df):
    """
    Affiche l'historique d'entraînement d'un modèle
    :param df: Dataframe correspondant à l'historique d'entraînement du modèle
    :return: None
    """
    plt.plot(df['age_mean_absolute_error'])
    plt.plot(df['val_age_mean_absolute_error'])
    plt.title("Age prédiction")
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend(['TRAIN', 'VALIDATION'], loc='upper right')
    plt.show()

    plt.plot(df['genre_binary_accuracy'])
    plt.plot(df['val_genre_binary_accuracy'])
    plt.title("Genre prédiction")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend(['TRAIN', 'VALIDATION'], loc='upper right')
    plt.show()

    plt.plot(df['ethnie_categorical_accuracy'])
    plt.plot(df['val_ethnie_categorical_accuracy'])
    plt.title("Ethnie prédiction")
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend(['TRAIN', 'VALIDATION'], loc='upper right')
    plt.show()



import pandas as pd
df = pd.read_csv("XCEPTION/History/history_model2.csv")
afficher_historique_d_entrainement(df)