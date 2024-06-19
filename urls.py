"""
URL configuration for site_authentification project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""


from django.contrib import admin
from django.urls import path
from auth_app import views
from django.conf.urls.static import static
from django.conf import settings
#from django.contrib.auth import views as auth_views




urlpatterns = [
    path('admin/', admin.site.urls),
    path('register/', views.register, name='register'),
    path('verification/', views.validate, name='validate'),
    path('accueil/', views.home, name='home'),
    path('profil/', views.profile, name='profile'),
    path('connexion/', views.connexion, name='connexion'),
    path('examen/', views.examen, name='examen'),
    path('enseignant/', views.enseignant, name='enseignant'),
    path('playlist/', views.playlist, name='playlist'),
    path('deconnexion/', views.deconnexion, name='deconnexion'),
    path('capture/', views.capture, name='capture'),
    path('login/', views.login, name='login'),
    path('upload_photo/', views.upload_photo, name='upload_photo'),
   # path("recognize/", views.recognize, name='recognize')
   # path('reconnaissance_faciale/', views.reconnaissance_faciale, name='reconnaissance_faciale'),
    path('indexe/', views.indexe, name='indexe'),
    path('video_feed/', views.video_feed, name='video_feed')
    
    
   

    
    #path('accounts/login/', auth_views.LoginView.as_view(), name='login'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)