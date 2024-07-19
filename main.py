"""
Auteur : AURELIEN PINON - PINA20100100
Fichier Main
"""
import cv2
import constante as CS
import numpy as np
from CNN import CNN
from VGG import AgeGenreEtEthniePredictionUtilisantVGG
from Xception import AgeGenreEtEthniePredictionUtilisantXception
from tkinter.filedialog import askopenfilename
from dataprep import preprocess_image
from keras.utils import to_categorical

# Main
if __name__ == '__main__':
    # Télécharge le modèle CNN
    cnn = CNN()
    cnn.load_model("CNN/model_final.h5")

    # Télécharge le modèle VGG
    vgg = AgeGenreEtEthniePredictionUtilisantVGG()
    vgg.load_model("VGG/model_final.h5")

    # Télécharge le modèle XCeption
    xception = AgeGenreEtEthniePredictionUtilisantXception()
    xception.load_model("XCEPTION/model_final.h5")

    # Interface utilisateur
    ARRET = False
    while not ARRET:
        CS.clear()
        # Récupération de l'image à traiter
        print("Veuillez sélectionner une image à traiter...\n")
        img_path = askopenfilename(initialdir="./", title="Selectionner une image",
                                   filetypes=[("Fichier jpeg", "*.jpg"), ("Fichier png", "*.png")])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extraction d'un visage de l'image
        img_test, etat_preprocessing = preprocess_image(img)

        if etat_preprocessing == CS.ECHEC: print(f"Aucun visage n'a été détecté, veuillez fournir une autre image !")
        else:
            # Demande à l'utilisateurs de rentrer les bonnes valeurs pour pouvoir évaluer chaque modèle
            information = input("Voulez-vous donner les informations concernant cette photo pour pouvoir "
                                "évaluer les modèles ?\n'y' pour oui : ")
            if information == 'y':
                age, genre, ethnie = 0, 0, 0
                while True:
                    try:
                        age = int(input("Entrez l'âge de la personne sur la photo : "))
                    except ValueError:
                        print("Veuillez rentrer une valeur valide")
                        continue
                    if age >= 1 and age <= CS.AGE_MAX: break
                    else: print("Veuillez rentrer un âge compris entre 1 et " + str(CS.AGE_MAX) + "ans")

                while True:
                    try:
                        genre = int(input("Entrez le genre de la personne sur la photo "
                                          "(0 pour un Homme 1 pour une Femme): "))
                    except ValueError:
                        print("Veuillez rentrer une valeur valide !")
                        continue
                    if genre == 0 or genre == 1: break
                    else: print("Veuillez rentrer soit 0 ou 1 !")

                while True:
                    try:
                        ethnie = int(input("Entrez l'ethnie' de la personne sur la photo "
                                           "(0 = Caucasien, 1 = Africain, 2 = Asiatique, 3 = Indien, 4 = Autre): "))
                    except ValueError:
                        print("Veuillez rentrer une valeur valide !")
                        continue
                    if ethnie >= 0 and ethnie <= 4: break
                    else:print("Veuillez rentrer une valeur comprise entre 0 et 4 !")
                CS.clear()

            # Prédictions des modèles
            pred_cnn = cnn.predict(img_test)
            pred_vgg = vgg.predict(img_test)
            pred_xception = xception.predict(img_test)

            # Affichage des prédictions
            print(f"Prédiction du modèle CNN : ")
            age_pred_cnn, genre_pred_cnn, ethnie_pred_cnn = cnn.decode_prediction(pred_cnn)
            print(f'    Age : ' + str(age_pred_cnn) + ' ans')
            print(f'    Genre : ' + str(genre_pred_cnn[0]) + ' ' + str(genre_pred_cnn[1]) + '%')
            print(f'    Ethnie : ' + str(ethnie_pred_cnn[0]) + ' ' + str(ethnie_pred_cnn[1]) + '%')

            print(f"\nPrédiction du modèle VGG : ")
            age_pred_vgg, genre_pred_vgg, ethnie_pred_vgg = vgg.decode_prediction(pred_vgg)
            print(f'    Age : ' + str(age_pred_vgg) + ' ans')
            print(f'    Genre : ' + str(genre_pred_vgg[0]) + ' ' + str(genre_pred_vgg[1]) + '%')
            print(f'    Ethnie : ' + str(ethnie_pred_vgg[0]) + ' ' + str(ethnie_pred_vgg[1]) + '%')

            print(f"\nPrédiction du modèle XCEPTION : ")
            age_pred_xception, genre_pred_xception, ethnie_pred_xception = xception.decode_prediction(pred_xception)
            print(f'    Age : ' + str(age_pred_xception) + ' ans')
            print(f'    Genre : ' + str(genre_pred_xception[0]) + ' ' + str(genre_pred_xception[1]) + '%')
            print(f'    Ethnie : ' + str(ethnie_pred_xception[0]) + ' ' + str(ethnie_pred_xception[1]) + '%')

            # Affichage des métriques si les informations ont été fournies
            if information == 'y':
                # CNN
                mae_age = abs(age - age_pred_cnn)
                if (genre_pred_cnn[0] == 'Homme' and genre == 0) or (genre_pred_cnn[0] == 'Femme' and genre == 1):
                    accuracy_genre = 1
                else:
                    accuracy_genre = 0
                if ethnie_pred_cnn[0] == 'Caucasien': ethnie_pred_cnn[0] = 0
                elif ethnie_pred_cnn[0] == 'Africain': ethnie_pred_cnn[0] = 1
                elif ethnie_pred_cnn[0] == 'Asiatique': ethnie_pred_cnn[0] = 2
                elif ethnie_pred_cnn[0] == 'Indien': ethnie_pred_cnn[0] = 3
                else: ethnie_pred_cnn[0] = 4

                if ethnie_pred_cnn[0] == ethnie: accuracy_ethnie = 1
                else: accuracy_ethnie = 0
                print(f"\nMétrique d'évaluation modèle CNN : ")
                print(f"    Age MAE et RMSE (équivalent pour une valeur) : " + str(mae_age))
                print(f"    Genre Accuracy : " + str(accuracy_genre))
                print(f"    Ethnie Accuracy : " + str(accuracy_ethnie))

                # VGG
                mae_age = abs(age - age_pred_vgg)
                if (genre_pred_vgg[0] == 'Homme' and genre == 0) or (genre_pred_vgg[0] == 'Femme' and genre == 1):
                    accuracy_genre = 1
                else:
                    accuracy_genre = 0
                if ethnie_pred_vgg[0] == 'Caucasien': ethnie_pred_vgg[0] = 0
                elif ethnie_pred_vgg[0] == 'Africain': ethnie_pred_vgg[0] = 1
                elif ethnie_pred_vgg[0] == 'Asiatique': ethnie_pred_vgg[0] = 2
                elif ethnie_pred_vgg[0] == 'Indien': ethnie_pred_vgg[0] = 3
                else:ethnie_pred_vgg[0] = 4

                if ethnie_pred_vgg[0] == ethnie: accuracy_ethnie = 1
                else: accuracy_ethnie = 0
                print(f"\n\nMétrique d'évaluation modèle VGG : ")
                print(f"    Age MAE : " + str(mae_age))
                print(f"    Genre Accuracy : " + str(accuracy_genre))
                print(f"    Ethnie Accuracy : " + str(accuracy_ethnie))

                # XCEPTION
                mae_age = abs(age - age_pred_xception)
                if (genre_pred_xception[0] == 'Homme' and genre == 0) or (genre_pred_xception[0] == 'Femme' and genre == 1):
                    accuracy_genre = 1
                else:
                    accuracy_genre = 0
                if ethnie_pred_xception[0] == 'Caucasien': ethnie_pred_xception[0] = 0
                elif ethnie_pred_xception[0] == 'Africain': ethnie_pred_xception[0] = 1
                elif ethnie_pred_xception[0] == 'Asiatique': ethnie_pred_xception[0] = 2
                elif ethnie_pred_xception[0] == 'Indien': ethnie_pred_xception[0] = 3
                else: ethnie_pred_xception[0] = 4

                if ethnie_pred_xception[0] == ethnie: accuracy_ethnie = 1
                else: accuracy_ethnie = 0
                print(f"\n\nMétrique d'évaluation modèle Xception : ")
                print(f"    Age MAE : " + str(mae_age))
                print(f"    Genre Accuracy : " + str(accuracy_genre))
                print(f"    Ethnie Accuracy : " + str(accuracy_ethnie))

        # Condition d'arrêt
        choix = input("\nVoulez vous continuer y pour oui sinon n'importe quoi d'autre : ")
        if choix != 'y': ARRET = True