"""
Auteur : AURELIEN PINON - PINA20100100
Fichier contenant les fonctions relatives à la préparation des images.
Le but étant de recentrer chaque image autour du visage de la personne et de les redimensionner au format 256*256*3
"""
import os
import cv2
import math
import dlib
import numpy as np
import random
import shutil
import constante as CS

from PIL import Image
from colorama import Fore
from alive_progress import alive_bar

############################################# Fonction principale #############################################

def prepare_data():
    """
    Fonction principale servant à préparer les images au futur traitement.
    Traite les images contenu dans DOSSIER_IMAGES_BRUTS et enregistre le
    prétraitement dans DOSSIER_IMAGES_PRETRAITEES
    :return: None
    """
    if os.path.exists(CS.DOSSIER_IMAGES_PRETRAITEES): shutil.rmtree(CS.DOSSIER_IMAGES_PRETRAITEES)
    os.makedirs(CS.DOSSIER_IMAGES_PRETRAITEES)

    # Création des détecteurs de visage et d'yeux de CV2
    detecteur_visage  = cv2.CascadeClassifier(CS.FRONTAL_FACE_DETECTOR)
    detecteur_yeux    = cv2.CascadeClassifier(CS.EYE_DETECTOR)

    # Création des détecteurs de la librairie DLIB
    detecteur_dlib = dlib.get_frontal_face_detector()
    predicteur_dlib = dlib.shape_predictor(CS.PREDICTOR)

    # Création de variables servant au comptage d'images prétraitées
    nb_reussite = nb_echec = 0
    with alive_bar(total=compter_nb_image(CS.DOSSIER_IMAGES_BRUTS), title="Pré-traitement d'image", theme='smooth') as bar:

        # Parcours des images bruts
        for fichier in os.listdir(CS.DOSSIER_IMAGES_BRUTS):
            # On veut que la suite ne s'éxécute que sur les fichiers de type image
            extension_fichier = os.path.splitext(fichier)[1]
            if extension_fichier == ".png" or extension_fichier == ".jpg" or extension_fichier == ".jpeg":
                # Affichage de l'avancement
                bar.text = f"Working on Image : {fichier} !"
                bar()
                if not os.path.isfile(CS.DOSSIER_IMAGES_PRETRAITEES + '/' + fichier):
                    # Ouverture de l'image
                    image = cv2.imread(CS.DOSSIER_IMAGES_BRUTS + '/' + fichier)

                    # Tentative de prétraitement avec DLIB
                    angle, visage, etat = get_angle_and_face_using_dlib(image, detecteur_dlib, predicteur_dlib)
                    if etat == CS.SUCCES:
                        # Traitement de l'image
                        image = align_crop_and_resize(image, angle, visage)
                        # Enregistrement de la nouvelle image
                        image.save(CS.DOSSIER_IMAGES_PRETRAITEES + '/' + fichier)
                        nb_reussite += 1

                    # La détection DLIB ayant échoué on test avec CV2
                    else:
                        angle, visage, etat = get_angle_and_face_using_cv2(image, detecteur_visage, detecteur_yeux)
                        if etat == CS.SUCCES:
                            # Traitement de l'image
                            image = align_crop_and_resize(image, angle, visage)
                            # Enregistrement de la nouvelle image
                            image.save(CS.DOSSIER_IMAGES_PRETRAITEES + '/' + fichier)
                            nb_reussite += 1
                        else: nb_echec += 1

                else: nb_reussite += 1

    print(f"{Fore.BLUE}Total d'images brut : " + str(nb_reussite + nb_echec))
    print(f"{Fore.GREEN}Total d'images prétraitées avec succès : " + str(nb_reussite))
    print(f"{Fore.RED}Total d'échec : " + str(nb_echec))

############################################## Prétraitement d'image ##############################################

def preprocess_image(image):
    """
    Fonction effectuant le preprocessing d'une image passée en paramètre
    :param image: Image sous forme de tableau
    :return: L'image prétraitée et normalisée, retourne l'état du preprocessing (SUCCES ou ECHEC)
    """
    # Création des détecteurs de visage et d'yeux de CV2
    detecteur_visage  = cv2.CascadeClassifier(CS.FRONTAL_FACE_DETECTOR)
    detecteur_yeux    = cv2.CascadeClassifier(CS.EYE_DETECTOR)

    # Création des détecteurs de la librairie DLIB
    detecteur_dlib = dlib.get_frontal_face_detector()
    predicteur_dlib = dlib.shape_predictor(CS.PREDICTOR)

    # Tentative de prétraitement avec DLIB
    angle, visage, etat = get_angle_and_face_using_dlib(image, detecteur_dlib, predicteur_dlib)
    if etat == CS.SUCCES:
        image = np.array(align_crop_and_resize(image, angle, visage))
        image = image / 255.0
        image = np.expand_dims(image, 0)
        return image, CS.SUCCES

    # La détection DLIB ayant échoué on test avec CV2
    angle, visage, etat = get_angle_and_face_using_cv2(image, detecteur_visage, detecteur_yeux)
    if etat == CS.SUCCES:
        image = np.array(align_crop_and_resize(image, angle, visage))
        image = image / 255.0
        image = np.expand_dims(image, 0)
        return image, CS.SUCCES

    return None, CS.ECHEC

############################################# Prétraitement avec DLIB #############################################

def get_angle_and_face_using_dlib(image, detecteur, predicteur):
    """
    Détermine l'angle de rotation et l'emplacement du visage à partir de la librairie DLIB
    :param image: Une image obtenu à partir de la librairie CV2
    :param detecteur: Détecteur de visage obtenu avec la librairie DLIB
    :param predicteur: Predit les points de repère d'un visage, obtenu avec la librairie DLIB
    :return: L'état de l'opération (SUCCES ou ECHEC) ainsi que l'angle de rotation
    et le rectangle entourant le visage (None en cas d'échec)
    """
    # Détermine la position du visage
    image_gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    visage = detecteur(image_gray, 0)

    # Vérifie l'état de la prédiction
    if len(visage) > 0:
        # Obtention des caractéristiques du visage
        x, y, largeur, hauteur = position_visage(visage[0])
        rectangle_visage = [x, y, largeur, hauteur]

        # Obtention des positions des yeux et du nez
        forme = predicteur(image_gray, visage[0])
        oeil_gauche, oeil_droit, nez = position_nez_yeux(forme)

        # Calcul du centre du front et du centre de haut de la prédiction
        centre_front = ((oeil_gauche[0] + oeil_droit[0]) // 2, (oeil_gauche[1] + oeil_droit[1]) // 2)
        centre_prediction = (int((x + largeur) / 2), y)

        # Calcul des distances caractéristiques formant un triangle sur le visage
        distance_front_nez        = distance_euclidienne(centre_front, nez)
        distance_prediction_nez   = distance_euclidienne(centre_prediction, nez)
        distance_prediction_front = distance_euclidienne(centre_prediction, centre_front)

        # Calcul de l'angle de rotation en utilisant la règle du cosinus cos(A) = (b^2+c^2-a^2) / 2bc
        cos_angle = (distance_prediction_nez ** 2 + distance_front_nez ** 2 - distance_prediction_front ** 2) \
                    / (2 * distance_prediction_nez * distance_front_nez)
        if cos_angle > 1: cos_angle=1
        elif cos_angle < -1: cos_angle=-1
        angle_radian = np.arccos(cos_angle)

        # Calcul du centre de rotation de l'image
        centre_rotation = calcul_centre_rotation(nez, centre_front, angle_radian)
        centre_rotation = (int(centre_rotation[0]), int(centre_rotation[1]))

        # Détermine l'angle en degrés
        if appartient_au_triangle(nez, centre_front, centre_prediction, centre_rotation):
            angle_degree = np.degrees(angle_radian)
        else:
            angle_degree = np.degrees(-angle_radian)

        return angle_degree, rectangle_visage, CS.SUCCES

    return None, None, CS.ECHEC

def position_visage(visage):
    """
    Extrait les positions caractéristiques de l'objet visage
    :param visage: Tableau obtenu à partir d'un détecteur de visage DLIB
    :return: Les positions caractéristiques du visage
    """
    return visage.left(), visage.top(), visage.right(), visage.bottom()

def position_nez_yeux(forme):
    """
    A partir des points de repères du visage calcul la position du nez et du centre des deux yeux
    :param forme: Type Shape, obtenu par le prédicteur DLIB
    :return: Position du nez et du centre des deux yeux
    """
    point = points_repere(forme)
    oeil_gauche_x = int(point[3][1][0] + point[2][1][0]) // 2
    oeil_gauche_y = int(point[3][1][1] + point[2][1][1]) // 2
    oeil_droit_x  = int(point[1][1][0] + point[0][1][0]) // 2
    oeil_droit_y  = int(point[1][1][1] + point[0][1][1]) // 2
    nez = point[4][1]

    return [oeil_gauche_x, oeil_gauche_y], [oeil_droit_x, oeil_droit_y], nez

def points_repere(forme):
    """
    Obtention des coordonnées des points de repère du visage
    :param forme: Type Shape, obtenu par le prédicteur DLIB
    :return: Un tableau contenant les points de repères du visage
            point[0] et point[1] contiennent les points relatifs à l'oeil droit
            point[2] et point[3] l'oeil gauche
            point[4] le nez
    """
    point = []
    for i in range(0, 5): point.append((i, (forme.part(i).x, forme.part(i).y)))

    return point

def calcul_centre_rotation(origine, point, angle):
    """
    Calcul du centre de rotation pour aligner deux points
    :param origine: Premier point
    :param point: Deuxième point
    :param angle: Angle en radian entre origine et point
    :return: Position du centre de rotation
    """
    ox, oy = origine
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def appartient_au_triangle(point1, point2, point3, extra_point):
    """
    Détermine si extra_point fais partie de l'aire du triangle
    :param point1: (x1, y1)
    :param point2: (x2, y2)
    :param point3: (x3, y3)
    :param extra_point: (epx, epy)
    :return: True si le point est dans le triangle, false sinon
    """
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0): return True
    return False

############################################# Prétraitement avec CV2 #############################################

def get_angle_and_face_using_cv2(image, detecteur_visage, detecteur_yeux):
    """
    Détermine l'angle de rotation et l'emplacement du visage à partir de la librairie CV2
    :param image: Une image obtenu à partir de la librairie CV2
    :param detecteur_visage: Détecteur de visage de type CV2_CascadeClassifier
    :param detecteur_yeux: Détecteur d'yeux de type CV2_CascadeClassifier
    :return: L'état de l'opération (SUCCES ou ECHEC) ainsi que l'angle de rotation
    et le rectangle entourant le visage (None en cas d'échec)
    """
    # Détection du visage à l'aide de CV2
    visage, detection = detection_visage_cv2(image, detecteur_visage)
    if detection == CS.ECHEC: return None, None, CS.ECHEC

    # Détection des yeux à l'aide de CV2
    yeux, detection = detection_yeux_cv2(image, detecteur_yeux)
    if detection == CS.ECHEC: return None, None, CS.ECHEC

    # Détermine l'angle de rotation de l'image
    angle = angle_rotation(yeux)

    return angle, visage, CS.SUCCES

def detection_visage_cv2(image, detecteur):
    """
    Tente de détecter un visage à partir d'une image
    :param image: Une image obtenu à partir de la librairie CV2
    :param detecteur: Détecteur de visage de type CV2_CascadeClassifier
    :return: En cas de Succès les coordonnées du rectangle entourant le visage.
    En cas d'échec un tableau contenant que des valeurs négatives
    L'état de la détection (SUCCES ou ECHEC)
    """
    visage = detecteur.detectMultiScale(image, 1.3, 5)
    if len(visage) > 0:
        visage_x, visage_y, visage_largeur, visage_hauteur = visage[0]
        return [visage_x, visage_y, visage_largeur, visage_hauteur], CS.SUCCES

    return [-1, -1, -1, -1], CS.ECHEC

def detection_yeux_cv2(image, detecteur):
    """
    Tente de détecter les yeux d'un visage à partir d'une image
    :param image: Une image obtenu à partir de la librairie CV2
    :param detecteur: Détecteur d'yeux de type CV2_CascadeClassifier
    :return: En cas de Succès un tableau contenant les positions des rectangles
    entourant les deux yeux.
    En cas d'échec un tableau contenant que des valeurs négatives
    L'état de la détection (SUCCES ou ECHEC)
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Le détecteur d'yeux nécessite une image en noir et blanc
    yeux = detecteur.detectMultiScale(image)
    if len(yeux) == 2:
        if yeux[0][0] < yeux[1][0]:
            oeil_gauche = yeux[0]
            oeil_droit  = yeux[1]
        else:
            oeil_gauche = yeux[1]
            oeil_droit = yeux[0]

        return [oeil_gauche, oeil_droit], CS.SUCCES

    return [[-1, -1, -1, -1], [-1, -1, -1, -1]], CS.ECHEC

def centre_yeux(yeux):
    """
    Calcul le centre des deux yeux d'un visage
    :param yeux: Matrice contenant les positions des deux yeux sous forme de rectangle
    :return: 4 valeurs contenant les positions x et y du centre des deux yeux
    """
    oeil_gauche_x = int(yeux[0][0] + (yeux[0][2] / 2))
    oeil_gauche_y = int(yeux[0][1] + (yeux[0][3] / 2))
    oeil_droit_x = int(yeux[1][0] + (yeux[1][2] / 2))
    oeil_droit_y = int(yeux[1][1] + (yeux[1][3] / 2))

    return oeil_gauche_x, oeil_gauche_y, oeil_droit_x, oeil_droit_y

def angle_rotation(yeux):
    """
    Calcul l'angle de rotation nécessaire pour aligner le visage en se basant sur
    la position des deux yeux
    :param yeux: Matrice contenant les positions des deux yeux sous forme de rectangle
    :return: L'angle de rotation nécessaire ainsi que sa direction (HORAIRE ou ANTIHORAIRE)
    """
    # Détermine les coordonnées du centre des deux yeux
    oeil_gauche_x, oeil_gauche_y, oeil_droit_x, oeil_droit_y = centre_yeux(yeux)

    if oeil_gauche_y > oeil_droit_y:
        # L'oeil gauche étant plus haut la rotation sera dans le sens anti_horaire
        angle_droit = (oeil_gauche_x, oeil_droit_y)
        direction = CS.ANTI_HORAIRE
    else:
        # L'oeil gauche étant plus bas la rotation sera dans le sens anti_horaire
        angle_droit = (oeil_droit_x, oeil_gauche_y)
        direction = CS.SENS_HORAIRE

    # Calcul la longueur des côtés du triangle rectangle
    adjacent = distance_euclidienne((oeil_droit_x, oeil_droit_y), angle_droit)
    hypotenus = distance_euclidienne((oeil_droit_x, oeil_droit_y), (oeil_gauche_x, oeil_gauche_y))

    # Calcul de l'angle
    cos_angle = adjacent/hypotenus
    angle_radian = np.arccos(cos_angle)
    angle_degree = (angle_radian*180) / math.pi

    if direction == -1: angle_degree = 90 - angle_degree

    return angle_degree * direction

############################### Application du prétraitement sur une image ###############################

def align_crop_and_resize(image, angle, position_visage):
    """
    Aligne, croppe et resize l'image
    :param image: Une image obtenu à partir de la librairie CV2 au format BGR
    :param angle: Angle de rotation pour l'alignement
    :param position_visage: Position du visage pour le crop
    :return: Retourne l'image alignée, croppée et redimensionnée
    """
    # Passage de l'image en canaux RGB
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Alignement de l'image
    image_alignee = Image.fromarray(imageRGB)
    image_alignee = np.array(image_alignee.rotate(angle))

    # Crop de l'image
    taille_image = imageRGB.shape[:2]
    crop_right = int(position_visage[0] + position_visage[2])
    crop_lower = int(position_visage[1] + position_visage[3])
    if crop_right > taille_image[1]: crop_right = taille_image[1]
    if crop_lower > taille_image[0]: crop_lower = taille_image[0]

    image_alignee = Image.fromarray(image_alignee)
    image_croppee = image_alignee.crop((int(position_visage[0]), int(position_visage[1]),
                                        crop_right, crop_lower))

    # Retourne l'image redimensionnée
    return image_croppee.resize(CS.NOUVEAU_FORMAT)

############################################ Fonctions auxiliaires ############################################

def distance_euclidienne(point_1, point_2):
    """
    Calcul la distance euclidienne entre deux points dans un plan 2D
    :param point_1: (x1, y1) coordonnées du point1
    :param point_2: (x2, y2) coordonnées du point2
    :return: La distance entre les deux points en float
    """
    return math.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)

def compter_nb_image(DOSSIER):
    """
    Compte le nombre d'image de type 'PNG' 'JPG' et 'JPEG' d'un dossier
    :param DOSSIER: Nom du dossier à traiter
    :return: Le nombre d'image contenu dans le dossier en int
    """
    nb_image = 0
    for fichier in os.listdir(DOSSIER):
        extension_fichier = os.path.splitext(fichier)[1]
        if extension_fichier == ".png" or extension_fichier == ".jpg" or extension_fichier == ".jpeg": nb_image += 1

    return nb_image

#################################### Fonction de vérification ###################################
def verification():
    """
    Vérifie la qualité des images prétraitées en tentant de détecter un visage avec les deux méthodes
    de détection CV2 et DLIB. Puis déplace les images dont le test n'est pas concluant dans un autre dossier
    :return: None
    """
    NOM_DOSSIER = "Images/Mauvais_pretraitement_dlib_et_cv2"
    if not os.path.exists(NOM_DOSSIER): os.makedirs(NOM_DOSSIER)

    # Création du détecteur de visage de CV2
    detecteur_visage  = cv2.CascadeClassifier(CS.FRONTAL_FACE_DETECTOR)

    # Création du détecteur de visage de la librairie DLIB
    detecteur_dlib = dlib.get_frontal_face_detector()

    nb_mauvais_traitement = 0
    with alive_bar(total=compter_nb_image(CS.DOSSIER_IMAGES_PRETRAITEES), title="Vérification des images",
                   theme='smooth') as bar:
        for file in os.listdir(CS.DOSSIER_IMAGES_PRETRAITEES):
            extension_fichier = os.path.splitext(file)[1]
            if extension_fichier == ".png" or extension_fichier == ".jpg" or extension_fichier == ".jpeg":
                # Affichage de l'avancement
                bar.text = f"Working on Image : {file} !"
                bar()
                # Ouverture de l'image
                image = cv2.imread(CS.DOSSIER_IMAGES_PRETRAITEES + '/' + file)
                # Détermine la position du visage avec DLIB
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                visage_dlib = detecteur_dlib(image_gray, 0)

                if len(visage_dlib) == 0:
                    visage_cv2, detection_cv2 = detection_visage_cv2(image, detecteur_visage)
                    if detection_cv2 == CS.ECHEC:
                        os.replace(CS.DOSSIER_IMAGES_PRETRAITEES + '/' + file, NOM_DOSSIER + '/' + file)
                        nb_mauvais_traitement += 1

    print(f"{Fore.BLUE}Total d'images brut : " + str(compter_nb_image(CS.DOSSIER_IMAGES_BRUTS)))
    print(f"{Fore.GREEN}Total d'images prétraitées avec succès : " + str(compter_nb_image(CS.DOSSIER_IMAGES_PRETRAITEES)))
    print(f"{Fore.RED}Total de mauvais prétraitement : " + str(nb_mauvais_traitement))


################################## Fonction de SPLIT #####################################
def split_train_valid_test(part_train=0.7, part_valid=0.1):
    """
    Effectue le split en ensemble d'entraînement, de test et de validation des images prétraitées
    Parcours le dossier contenant les images prétraitées et copie de manière aléatoire les images
    dans les dossiers TRAIN TEST et VALIDATION. Le partitionnement se fait selon les taux passés
    en paramètre.
    :param part_train: Entre 0 et 1 (par défaut 70%). Part des données utilisées pour l'entraînement
    :param part_valid: Entre 0 et 1 (par défaut 10%). Part des données utilisées pour la validation
    :return: None
    """
    # Supprime les anciens dossiers s'ils existent
    if os.path.exists(CS.DOSSIER_TRAIN):      shutil.rmtree(CS.DOSSIER_TRAIN)
    if os.path.exists(CS.DOSSIER_TEST):       shutil.rmtree(CS.DOSSIER_TEST)
    if os.path.exists(CS.DOSSIER_VALIDATION): shutil.rmtree(CS.DOSSIER_VALIDATION)

    os.makedirs(CS.DOSSIER_TRAIN)
    os.makedirs(CS.DOSSIER_TEST)
    if part_valid > 0: os.makedirs(CS.DOSSIER_VALIDATION)

    list_image = os.listdir(CS.DOSSIER_IMAGES_PRETRAITEES)
    random.shuffle(list_image)

    i=0
    total_image = len(list_image)
    # Décommenter pour réduire le max de données à 1000 (sers pour les tests)
    #if total_image > 1000: total_image=1000; list_image = list_image[:1000]
    with alive_bar(total=total_image, title="SPLIT TRAIN-TEST-VALIDATION", theme='smooth') as bar:
        for img in list_image:
            # Affichage de l'avancement
            bar()
            # Copie des images dans les dossiers TRAIN et TEST
            chemin_source = CS.DOSSIER_IMAGES_PRETRAITEES + '/' + img
            if i < total_image * part_train: shutil.copyfile(chemin_source, CS.DOSSIER_TRAIN + '/' + img)
            elif i >= total_image * part_train and i < total_image * (1 - part_valid):
                shutil.copyfile(chemin_source, CS.DOSSIER_TEST + "/" + img)
            else: shutil.copyfile(chemin_source, CS.DOSSIER_VALIDATION + '/' + img)
            i += 1