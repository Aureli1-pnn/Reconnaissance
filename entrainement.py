"""
Auteur : AURELIEN PINON - PINA20100100
Fichier contenant les fonctions relatives à l'entraînement et l'évaluation des différents modèles
"""
import os
import pandas as pd
import numpy as np
import constante as CS
from extraction import get_dataframe
from CNN import CNN
from VGG import AgeGenreEtEthniePredictionUtilisantVGG
from extraction import generer_entrer
from Xception import AgeGenreEtEthniePredictionUtilisantXception

# Créer un dossier qui contiendra les poids du modèle lors de l'entraînement du CNN
if not os.path.exists("CNN/Weights/Model" + str(CS.NUMERO_MODEL_CNN)):
    os.makedirs("CNN/Weights/Model" + str(CS.NUMERO_MODEL_CNN))


# Créer un dossier qui contiendra les poids du modèle lors de l'entraînement du VGG
if not os.path.exists("VGG/Weights/Model" + str(CS.NUMERO_MODEL_VGG)):
    os.makedirs("VGG/Weights/Model" + str(CS.NUMERO_MODEL_VGG))

# Créer un dossier qui contiendra les poids du modèle lors de l'entraînement du XCEPTION
if not os.path.exists("XCEPTION/Weights/Model" + str(CS.NUMERO_MODEL_XCEPTION)):
    os.makedirs("XCEPTION/Weights/Model" + str(CS.NUMERO_MODEL_XCEPTION))

# Création des dataframes
df_train = get_dataframe(CS.DOSSIER_TRAIN, type='Train')
df_valid = get_dataframe(CS.DOSSIER_VALIDATION, type="Valid")
df_test  = get_dataframe(CS.DOSSIER_TEST, type='Test ')

# Création des générateurs de données
train_gen = generer_entrer(df_train, True, CS.BATCH_SIZE)
valid_gen = generer_entrer(df_valid, True, CS.BATCH_SIZE)
test_gen  = generer_entrer(df_test, True, CS.BATCH_SIZE)

def entrainement_cnn():
    # Création du modèle CNN
    cnn = CNN()

    # Entraînement du modèle

    history = cnn.train(train_gen, nb_donnees_train=len(df_train), nb_donnees_valid=len(df_valid), epochs=CS.EPOCHS,
                        batch_size=CS.BATCH_SIZE, callbacks=[CS.CHECKPOINT_CALLBACK_CNN], valid_data=valid_gen,
                        verbose=1)

    # Enregistrement du modèle
    cnn.save_model("CNN/model" + str(CS.NUMERO_MODEL_CNN) + ".h5")

    # Obtient les statistiques de l'entraînement
    hist_df = pd.DataFrame(history.history)

    # Enregistre les statistiques de l'entraînement
    hist_json_file = 'CNN/History/history_model' + str(CS.NUMERO_MODEL_CNN) + '.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    hist_csv_file = 'CNN/History/history_model' + str(CS.NUMERO_MODEL_CNN) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

def entrainer_vgg():
    # Création du modèle VGG
    vgg = AgeGenreEtEthniePredictionUtilisantVGG()

    # Entraînement du modèle
    history = vgg.train(train_gen, nb_donnees_train=len(df_train), nb_donnees_valid=len(df_valid), epochs=CS.EPOCHS,
                        batch_size=CS.BATCH_SIZE, callbacks=[CS.CHECKPOINT_CALLBACK_VGG], valid_data=valid_gen,
                        verbose=2)

    # Enregistrement du modèle
    vgg.save_model("VGG/model" + str(CS.NUMERO_MODEL_VGG) + ".h5")

    # Obtient les statistiques de l'entraînement
    hist_df = pd.DataFrame(history.history)

    # Enregistre les statistiques de l'entraînement
    hist_json_file = 'VGG/History/history_model' + str(CS.NUMERO_MODEL_CNN) + '.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    hist_csv_file = 'VGG/History/history_model' + str(CS.NUMERO_MODEL_CNN) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

def entrainer_Xception():
    # Création du modèle Xception
    Xception = AgeGenreEtEthniePredictionUtilisantXception()
    Xception.summary()

    # Entraînement du modèle
    history = Xception.train(train_gen, nb_donnees_train=len(df_train), nb_donnees_valid=len(df_valid), epochs=CS.EPOCHS,
                        batch_size=CS.BATCH_SIZE, callbacks=[CS.CHECKPOINT_CALLBACK_XCEPTION], valid_data=valid_gen,
                        verbose=1)

    # Enregistrement du modèle
    Xception.save_model("XCEPTION/model" + str(CS.NUMERO_MODEL_XCEPTION) + ".h5")

    # Obtient les statistiques de l'entraînement
    hist_df = pd.DataFrame(history.history)

    # Enregistre les statistiques de l'entraînement
    hist_json_file = 'XCEPTION/History/history_model' + str(CS.NUMERO_MODEL_XCEPTION) + '.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    hist_csv_file = 'XCEPTION/History/history_model' + str(CS.NUMERO_MODEL_XCEPTION) + '.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

def evaluation_model(model_type='CNN'):
    """
    Procède à l'évaluation d'un modèle sur l'ensemble de test
    :param model_type: Type de modèle à évaluer
    :return: None
    """
    if model_type=='CNN':
        model = CNN()
        model.load_model("CNN/model_final.h5")
        print("\nEvaluation model CNN : ")
    elif model_type=='VGG':
        model = AgeGenreEtEthniePredictionUtilisantVGG()
        model.load_model("VGG/model_final.h5")
        print("\nEvaluation model VGG : ")
    else:
        model = AgeGenreEtEthniePredictionUtilisantXception()
        model.load_model("XCEPTION/model_final.h5")
        print("\nEvaluation model Xception : ")

    eval = model.evaluate(test_gen, verbose=1, taille_test=len(df_test))

    loss_total, loss_age, loss_genre, loss_ethnie = eval[0], eval[1], eval[2], eval[3]
    mae_age, rmse_age = eval[4], eval[5]
    accuracy_genre = eval[6]
    accuracy_ethnie = eval[7]
    print(f"\nLoss : " + str(loss_total))
    print(f"Loss Age: " + str(loss_age))
    print(f"Loss Genre: " + str(loss_genre))
    print(f"Loss Ethnie: " + str(loss_ethnie) + "\n")
    print(f"Age MAE : " + str(mae_age))
    print(f"Age RMSE : " + str(rmse_age))
    print(f"Genre Accuracy : " + str(accuracy_genre))
    print(f"Ethnie Accuracy : " + str(accuracy_ethnie))

    # Enregistrement de l'évaluation dans un fichier
    dict_eval = {'Loss total': loss_total, 'Loss Age': loss_age, 'Loss Genre': loss_genre, 'Loss_ethnie': loss_ethnie,
                 'MAE Age': mae_age, 'RMSE Age': rmse_age, 'Genre Accuracy': accuracy_genre,
                 'Ethnie Accuracy': accuracy_ethnie}

    if model_type=='CNN':   np.save("Evaluation/evaluation_cnn.npy", dict_eval)
    elif model_type=='VGG': np.save("Evaluation/evaluation_vgg.npy", dict_eval)
    else:                   np.save("Evaluation/evaluation_xception.npy", dict_eval)