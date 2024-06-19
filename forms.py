from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.core.mail import send_mail
from django.utils.crypto import get_random_string
from .models import CustomUser,Code


class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = CustomUser
        fields = ('email','image', 'password1', 'password2')

    

    def send_validation_code(self, user):
        code = get_random_string(length=6)
        code_obj = Code.objects.create(user=user, email=user.email, code=code)
        code_obj.save()

        subject = "Code de validation de l'inscription"
        message = f"Votre code de validation est : {code}"
        send_mail(subject, message, 'noreply@example.com', [user.email])
        
        
        

    def save(self, commit=True):
        
        user = super().save(commit=False)
        user.is_active = False  # Désactiver l'utilisateur jusqu'à la validation
        user.save()

        self.send_validation_code(user)

        return user



class ValidationForm(forms.Form):

    code = forms.CharField(max_length=6)

class ConnexionForm(forms.Form):
    email = forms.EmailField()
    mot_de_passe = forms.CharField(widget=forms.PasswordInput)


