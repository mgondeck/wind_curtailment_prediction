from django.urls import path, include

urlpatterns = [
    path('', include('myapp.urls')),
#    path('', include('wind_curtailment.urls')),
]