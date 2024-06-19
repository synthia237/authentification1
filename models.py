
# Create your models here.

from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.db import models
from django.core.files.base import ContentFile
from django.core.files.uploadedfile import SimpleUploadedFile


class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, nom=None, image=None):
        if not email:
            raise ValueError("L'adresse e-mail est requise")

        user = self.model(email=self.normalize_email(email), nom=nom, image=image)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password, nom=None, image=None):
        user = self.create_user(email, password, nom, image)
        user.is_admin = True
        user.save(using=self._db)
        return user


class CustomUser(AbstractBaseUser):
    email = models.EmailField(max_length=255, unique=True)
    id = models.AutoField(primary_key=True)
    image = models.ImageField(upload_to='site_authentification/auth_app/static/images', blank=True, null=True,)
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)
    class Meta:
        ordering = ['id']


    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def has_perm(self, perm, obj=None):
        return True

    def has_module_perms(self, app_label):
        return True

    @property
    def is_staff(self):
        return self.email




class Code(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, default=72)
    email = models.EmailField()
    code = models.CharField(max_length=6)
    id = models.AutoField(primary_key=True)

    
    def __str__(self):
        return self.code






