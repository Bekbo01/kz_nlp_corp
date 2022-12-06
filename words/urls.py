from django.urls import path

from . import views

urlpatterns = [
    path('q/<str:text>/', views.index, name='index'),
]
