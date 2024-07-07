from django.contrib.auth import login
from django.shortcuts import redirect, render
from .forms import RegistrationForm,ValidationForm,ConnexionForm
from django.contrib.auth.decorators import login_required, permission_required
from .models import CustomUser
from django.utils.crypto import get_random_string
from .models import Code
import cv2
import os
import face_recognition
from django.conf import settings
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import io
import base64
from PIL import Image
from django.http import HttpResponse
import urllib.request
import shutil
from django.http import StreamingHttpResponse, HttpResponseServerError
from django.views.decorators import gzip
from django.contrib.auth.hashers import check_password





from django.conf import settings




def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST, request.FILES) or None
        
        if form.is_valid():
            email=form.cleaned_data['email']
            image= form.cleaned_data['image']
            user_profile = form.save(commit=False)
            user_profile.image = image.name  # Ou utilisez un nom unique pour l'image
            user_profile.save()
            # Créer le chemin du dossier à partir de l'e-mail
            folder_name = email.replace("@", "_").replace(".", "_")

            # Chemin complet du dossier d'images
            image_directory = os.path.join(settings.MEDIA_ROOT, folder_name)

            # Créer le dossier s'il n'existe pas déjà
            if not os.path.exists(image_directory):
             try:
              os.makedirs(image_directory)
             except OSError as e:
              print(f"Erreur lors de la création du dossier : {e}")      
            # Enregistrer l'image dans le dossier
            image_file = request.FILES['image']
            fs = FileSystemStorage(location=image_directory)
            fs.save(image_file.name, image_file)
            return redirect('sucess')
    else:
        form = RegistrationForm()

    return render(request, 'register.html', {'form': form})




from django.shortcuts import render, redirect
from django.contrib import messages

def validate(request):
    error_message= None
    if request.method == 'POST':
        form = ValidationForm(request.POST)
        if form.is_valid():
            code = form.cleaned_data['code']
            # Check if the code already exists in the database for this email
            code_obj = Code.objects.filter(code=code).first()
            if code_obj:
                # Code already exists for this email, update the existing code
                code_obj.code = code
                return redirect('home')
            else:
                # Code does not exist, create a new Code object and save it to the database
                
                error_message = 'Le code est incorrect. Veuillez reessayer!'
    else:
        form = ValidationForm()

    return render(request, 'validate.html', {'form': form, 'error_message': error_message})



def capture(request):
    return render(request, 'capture.html')
'''
def recognize(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST) 
        photo = 'site_authentification/auth_app/IMAGE/PHOTO.png'
        if form.is_valid():
           
            image = form.cleaned_data['image']
            try:
                known_image = face_recognition.load_image_file(photo)
                unknown_image = face_recognition.load_image_file(image)

                amy_encoding = face_recognition.face_encodings(known_image)[0]
                unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
                results = face_recognition.compare_faces([amy_encoding], unknown_encoding)

                if results[0]:
                    return JsonResponse({'message': 'Match found'})  # Correspondance trouvée
                else:
                    return JsonResponse({'message': 'No match found'})  # Aucune correspondance trouvée

            except Exception as e:
                return JsonResponse({'error': str(e)}, status=500)  # Gestion des erreurs lors du traitement des images
        else:
            return JsonResponse({'error': 'Le formulaire n\'est pas valide.'})
    else:
        return JsonResponse({'error': 'La méthode HTTP doit être POST.'}, status=400)
 '''       



def ecran(request):
    return render(request, 'capt.html')


def home(request):
    return render(request, 'home.html')

def profile(request):
    return render(request, "profile.html")

#@csrf_exempt
def examen(request):
    """
    # Function to load images and encodings from the training images directory
    def load_known_images():
            known_encodings = []
            known_names = []

        # Path to the training images directory
            training_images_dir = os.path.join('media')

        # Loop through each subdirectory in the training images directory
            for person_dir in os.listdir(training_images_dir):
                person_name = person_dir if request.method == 'POST':
        form=RegistrationForm(request.POST)

            # Loop through each image in the person's subdirectory
            for image_file in os.listdir(os.path.join(training_images_dir, person_dir)):
                image_path = os.path.join(training_images_dir, person_dir, image_file)

                # Load the image and compute the face encoding
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]

                # Append the encoding and name to the lists
                known_encodings.append(encoding)
                known_names.append(person_name)

    return known_encodings, known_names

    # Load the known images and encodings
    known_encodings, known_names = load_known_images()

    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)

    # Load the Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using the Haar cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through each face found in the frame
        for (x, y, w, h) in faces:
            # Crop the face region from the frame
            face_image = frame[y:y+h, x:x+w]

            # Resize the face image for better recognition performance
            face_image = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)

            # Find the face encodings in the resized face image
            face_encodings = face_recognition.face_encodings(face_image)

            # Loop through each face encoding found in the resized face image
            for face_encoding in face_encodings:
                # Compare the face encoding with the known encodings
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                # If there is a match, find the index in known_encodings
                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]

                # Draw a rectangle around the face and display the name
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Facial Recognition', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all windows
        video_capture.release()
        cv2.destroyAllWindows() 
        """
        
    return render(request, 'examens.html')
    
    

def enseignant(request):
    return render(request, "enseignants.html")

def playlist(request):
    return render(request, "playlist.html")

def deconnexion(request):
    return redirect('connexion')

def connexion(request):
    if request.method == 'POST':
        form = ConnexionForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data.get('email')
            password = form.cleaned_data.get('mot_de_passe')
            
            customUser = CustomUser.objects.filter(email=email).first()
            if customUser and check_password(password, customUser.password):
                # Logique de connexion ici
                return redirect('home')
            else:
                form.add_error(None, "Email ou mot de passe incorrect.")
    else:
        form = ConnexionForm()
    
    return render(request, 'connexion.html', {'form': form})


def upload_photo(request):
    if request.method == 'POST':
        # Vérifier si le fichier est bien présent dans la requête
        if 'photo' in request.FILES:
            photo = request.FILES['photo']
            # Faites ici les opérations nécessaires pour enregistrer l'image sur le serveur
            # Par exemple, vous pouvez utiliser la méthode save() pour l'enregistrer dans un dossier spécifié
            filename = os.path.join(settings.BASE_DIR, 'auth_app', 'IMAGE', photo.name)
            # Écrire le contenu de l'image dans le fichier sur le serveur
            with open(filename, 'wb') as f:
                for chunk in photo.chunks():
                    f.write(chunk)
            
            # Retourner une réponse JSON indiquant que l'image a été téléchargée avec succès
            return JsonResponse({'message': 'Image téléchargée avec succès !'})
        
        # Si aucun fichier n'a été trouvé dans la requête
        return JsonResponse({'error': 'Aucun fichier trouvé dans la requête.'}, status=400)
    
    # Si la requête n'est pas de type POST
    return JsonResponse({'error': 'La méthode HTTP doit être POST.'}, status=400)



# Charger le classificateur Haar Cascade pour la détection de visages
def load_known_faces():
    known_encodings = []
    known_names = []

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

    return known_encodings, known_names
# Chargement des visages connus au démarrage du serveur
known_encodings, known_names = load_known_faces()

# Initialisation de la capture vidéo avec OpenCV
cap = cv2.VideoCapture(1)

# Initialisation du classificateur de détection de visages Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # Convertir l'image en niveaux de gris pour la détection de visages avec Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Détection de visages avec Haar Cascade
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Parcourir chaque visage détecté
            for (x, y, w, h) in faces:
                # Récupérer la région du visage
                face_image = frame[y:y+h, x:x+w]
                # Redimensionner l'image du visage pour la reconnaissance faciale
                small_face_image = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)
                
                # Obtenir les encodages du visage avec face_recognition
                face_encodings = face_recognition.face_encodings(small_face_image)
                
                # Comparaison des encodages avec les visages connus
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    name = "Inconnu"
                    
                    # Trouver la correspondance
                    if True in matches:
                        match_index = matches.index(True)
                        name = known_names[match_index]
                        
                         
                    # Dessiner un rectangle autour du visage et afficher le nom
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    
            # Conversion de l'image en format JPEG pour le streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Utilisation de gzip pour compresser les données vidéo en streaming
@gzip.gzip_page
def video_feed(request):
    try:
        return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
    except HttpResponseServerError as e:
        print("HTTP Response Error")

def indexe(request):
    return render(request, 'index.html')