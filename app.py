from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import face_recognition
import os

app = Flask(__name__)

# Chargement des visages connus au démarrage du serveur
known_encodings = []
known_names = []

# Fonction pour charger les visages connus depuis les fichiers d'image
def load_known_faces():
    global known_encodings, known_names
    
    # Chemin vers le répertoire contenant les images d'entraînement
    training_images_dir = 'media'  # Assurez-vous que c'est le chemin correct

    # Parcourir chaque sous-répertoire dans le répertoire des images d'entraînement
    for person_dir in os.listdir(training_images_dir):
        person_name = person_dir  # Utilisez le nom du répertoire comme nom de la personne
        for image_file in os.listdir(os.path.join(training_images_dir, person_dir)):
            image_path = os.path.join(training_images_dir, person_dir, image_file)
            
            # Vérifier si c'est bien un fichier (et non un répertoire)
            if os.path.isfile(image_path):
                # Charger l'image et calculer l'encodage facial
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]  # Prendre le premier visage trouvé
                known_encodings.append(encoding)
                known_names.append(person_name)

# Chargement des visages connus au démarrage du serveur
load_known_faces()

# Route pour la reconnaissance faciale
@app.route('/reconnaissance_faciale', methods=['POST'])
def reconnaissance_faciale():
    # Obtenir les données d'image encodées en base64 depuis la requête
    image_data = request.json.get('image_data', None)

    if image_data:
        # Décoder l'image depuis base64 et la convertir en tableau numpy
        image_data = base64.b64decode(image_data.split(',')[1])
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Détection des visages dans l'image avec face_recognition
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Comparaison des encodages avec les visages connus
        match_found = False
        recognized_name = "Inconnu"

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            if True in matches:
                match_index = matches.index(True)
                recognized_name = known_names[match_index]
                match_found = True
                break

        # Retourner la réponse sous forme de JSON
        return jsonify({
            'match_found': match_found,
            'recognized_name': recognized_name
        })
    else:
        return jsonify({
            'error': 'Aucune donnée d\'image reçue.'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Mettez à jour le port si nécessaire
