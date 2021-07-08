from django.contrib import admin

# Register your models here.
from django.contrib import admin
from . import models

admin.site.register(models.Image)

## TFServing 시작을 위한 도커 명령
# docker run -it --rm  \
#   -p 8500:8500 -p 8501:8501 \
#   --name detector \
#   -v /tmp/tensorboard:/tmp/tensorboard \
#   -v /home/ubuntu/myproj/serving/models/detector:/models/detector \
#   -e MODEL_NAME=detector \
#   tensorflow/serving