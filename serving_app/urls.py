from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', views.index, name='index'),
    path('upload', views.upload, name='upload'),
    path('predict', views.predict, name='predict'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)