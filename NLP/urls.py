"""NLP URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from recognition_service.views import *
from generation_service.views import *


urlpatterns = [
    url(r'rest-api', include('rest_framework.urls')),
    #url(r'rest-swagger/', include('rest_framework_swagger.urls')),
    url(r'^admin', include(admin.site.urls)),

    # recognition
    url(r'^recognize', RecognitionAPI.as_view()),

    # generation
    url(r'^generate', GenerationAPI.as_view()),
]
