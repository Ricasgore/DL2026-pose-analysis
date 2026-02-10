# Method_Automatic_pose_recognition2026

The goal of our project is to try to predict someone’s pose in a real time video with the correct label.

## Louis (UCF101 + ResNet3D-18)

    # process dataset 
        > prepocess_frames_lightLouis.py enables to convert dataset video into frames (.jpg images, size 24*24, quality 60) which are much lighter to process for the algorithm and therefore much faster for it to run. The program creates a target file called frames_img_224 where all the frames can be found.

        > datasetLouis.py randomly (shuffle=True) selects frames from frames_img_224 and chops them so that a video is uniformly covered by 16 frames. It gives out the corresponding label , it's just a sanity check. The selected frames can be seen in the file mes_frames_video in order to visually assess that the label is coherent.

    # train/test/validate 
        > main-vis-baselineLouis.py manages the training, testing and prediction using ResNet3D-18 on the previous dataset. For testing, a confusion matrix is generated (matrice_confusion_UCF_Louis.png).

    # predict_videoLouis.py uses ResNet3D-18 without training on UCF101 so useless.

## Kaïs
### CNN+LSTM trained on UCF101 dataset

I reused the CNN model from last year and add an LSTM taken from Stack Overflow, with some adjustements made.
UCF101 has been preprocess by Louis.

To train the model, launch main-vis-baselineKaisUCF.py with operation=0. You can change the batch size, the learning rate and the number of epochs.
To do the training in background, you can open a screen with: screen -S [name]
Then launch the script. You can leave it with Ctrl+A then D.
To list all of open screens: screen -ls
To attach to a created screen: screen -r [id.name]
To test the model, two options: 
- in order to have metrics (accuracy, MSE) of the model: launch main-vis-baselineKaisUCF.py with operation=1
- in order to test on a single video (have the 5 labels with hightest scores): launch main-vis-baselineKaisUCF.py with operation=4
To plot the loss during training: launch main-vis-baselineKaisUCF.py with operation=2

### Activity Net

I also tried to import an other video classification dataset ActivityNet in order to have more label classes (~200)
ActivityNet dataset is a json files (/home/amine_tsp/DL2026/Datasets/ActivityNet/Evaluation/data/activity_net.v1-3.min.json) with an Youtube URL, a start and end timecode and a label. I used the package yt-dlp to download the raw videos from the Youtube URLs (about 90% of the URL aren't working anymore). Then, I crop the raw videos from timecode data in order to extract clips of the action from the videos. Downloading took times but I now finished, you can try to process this dataset and adapt model and main for it.

## Matisse 

L'objectif de mes travaux était de concevoir un système capable de reconnaître des actions humaines non seulement sur des images fixes, mais surtout sur des séquences vidéo complexes provenant de différents types de sources (UCF101 et ActivityNet).
1. Gestion et Préparation des Données

    datasetMatisseUCF : Ce script sert de "lecteur" spécifique pour le dataset UCF101. Son rôle est d'aller chercher 16 images précises dans chaque vidéo et de les préparer pour que l'IA puisse les regarder. Il s'assure que toutes les images ont la même taille et les transforme en données numériques.

    datasetMatisse (ActivityNet) : Contrairement à UCF101, ActivityNet est un dataset beaucoup plus vaste et complexe. Ce script a été créé pour lire les fichiers d'annotations (format JSON) et retrouver les vidéos correspondantes dans une arborescence différente. Il sert de "traducteur" pour que le modèle puisse comprendre ce dataset spécifique.

2. Entraînement et Logique de Classification

    main-baselineMatisse : C'est le moteur principal. Il contient la logique pour entraîner le modèle sur des images simples. Il gère l'apprentissage (comment le modèle corrige ses erreurs) et la sauvegarde de son "cerveau" (le fichier .ckpt).

    main-baselineMatisseUCF : Une version adaptée du moteur précédent pour la vidéo. Il fait le lien entre ton lecteur d'images UCF101 et ton modèle pour apprendre à reconnaître les 101 catégories d'actions du dataset.

    main-baselineANMatisse (ActivityNet) : Pourquoi avoir créé ce script séparé ? ActivityNet ne se comporte pas comme UCF101 : le nombre de classes est différent (200 au lieu de 101), le format des étiquettes change, et les fichiers sont beaucoup plus lourds. Ce script a été nécessaire pour adapter la gestion de la mémoire (batch size) et la manière de mesurer la précision sans casser ce qui fonctionnait déjà pour UCF101. C'est le script "poids lourd" pour les entraînements de grande envergure.

3. Modélisation et Intelligence

    modelsMatisse : C'est la bibliothèque où je stocke les différentes structures de "cerveaux" IA.

        On y trouve le modèle FC (simple mais limité) pour les premiers tests.

        On y trouve le modèle CNN (plus évolué) qui est capable de détecter des formes et des silhouettes, indispensable pour la reconnaissance humaine.

    predictimageMatisse : Ce script est l'outil final de vérification. Une fois que le modèle est entraîné, on lui donne une vidéo qu'il n'a jamais vue, et il nous donne son verdict (ex: "Basketball à 95%"). C'est ici qu'on vérifie concrètement si l'IA a bien appris ou si elle se trompe.

4. Analyse des Résultats

    Visualisation (Plots et Matrices) : À la fin de chaque entraînement, des graphiques sont générés. La "Loss" permet de voir si l'IA progresse, tandis que la Matrice de Confusion nous montre si le modèle confond deux actions similaires (par exemple, confondre "Marcher" et "Courir").

## Alex 

L'objectif était de partir d'une reconnaissance d'image statique vers une classification des mouvements.

1. Entraînement des Modèles (Training)

    ResNet50 sur Dataset 1 : * Script : main-vis-baseline-Alex.py utilisant datasetAlex.py.

        Résultat : Précision médiocre. Le modèle traite les images de manière isolée et ne comprend pas la dynamique d'une action.

    ResNet50 + LSTM sur UCF101 :

        Script : main-vis-baseline-UCF-Alex.py utilisant datasetMatisseUCF.py.

        Résultat : Bonne précision (~72%). L'ajout de la couche LSTM permet au modèle de mémoriser l'enchaînement des frames pour identifier une action sur la durée.

2. Inférence en Temps Réel (Live Inference)

L'implémentation de l'inférence live a été faite sur MacBook en utilisant la webcam :

    Reconnaissance Humaine (ResNet50) : * Script : Human_detection.py (Dataset 2).

        Constat : Résultats aléatoires. Le modèle est trop sensible au décor et ne se concentre pas assez sur l'utilisateur.

    Reconnaissance d'Action (ResNet50 + LSTM) :

        Script : Action_recognition.py (UCF101).

        Constat : Bien plus stable. Le modèle identifie les actions (ex: Basketball, Punch) grâce à la répétition des mouvements, même si des erreurs persistent.

3. Traitement des Keypoints (OpenPose)

Pour dépasser les limites du flux RGB (couleurs/pixels), nous avons intégré OpenPose :

    Génération de Keypoints : Utilisation de openpose_compare_Alex.py pour extraire les squelettes des vidéos de UCF101.

    Vérification : Le script openpose_verification.py permet de valider visuellement que les points clés (articulations) sont correctement placés avant de les fournir au modèle.

    Note pour la suite : Une fois les keypoints extraits et vérifiés, l'étape ultime consiste à fusionner le flux ResNet50+LSTM et le flux Keypoints via une méthode de Two-Stream Fusion pour obtenir la précision maximale (bien que cela ralentisse l'inférence en raison du calcul de la pose).