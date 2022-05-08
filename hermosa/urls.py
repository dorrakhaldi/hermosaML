"""hermosa URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
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
from sitehermosa import views

urlpatterns = [

    path('', views.index, name='homepage'),
    path('burn', views.burnOut, name='burnpage'),
    path('burn1', views.burn1, name='burn1'),
    path('burn2', views.burn2, name='burn2'),
    path('burn3', views.burn3, name='burn3'),
    path('stress', views.stress, name='stress'),
    path('stress_form1', views.stress_form1, name='stress_form1'),
    path('stress_form2', views.stress_form2, name='stress_form2'),
    path('divorcetest1',views.divtest1 ,name='divorcetest1'),
    path('divorcetest2',views.divtest2 ,name='divorcetest2'),
    path('divorcetest3',views.divtest3 ,name='divorcetest3'),
    path('divorcetest4',views.divtest4 ,name='divorcetest4'),
    path('divorcetest5',views.divtest5 ,name='divorcetest5'),
    path('divorcetest6',views.divtest6 ,name='divorcetest6'),
    path('admin/', admin.site.urls),
]
