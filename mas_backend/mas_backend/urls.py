"""mas_backend URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
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
from mas_backend import views

urlpatterns = [
    path("admin/", admin.site.urls),

    # path('', views.home, name='home'),
    path('get_tic_tac_toe_action/', views.get_tic_tac_toe_action, name='get_tic_tac_toe_action'),
    path('get_pong_action/', views.get_pong_action, name='get_pong_action'),
    path('get_connect_4_action/', views.get_connect_4_action, name='get_connect_4_action'),

    path('reset_snake_game/', views.reset_snake_game, name='reset_snake_game'),
    path('get_snake_action/', views.get_snake_action, name='get_snake_action'),
]


