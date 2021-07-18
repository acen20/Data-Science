from django.shortcuts import render
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
import pickle
from .models import Features

# Create your views here

def home(request):
    #model = pickle.load(open('RBF.sav', 'rb'))
    with open('../RBF.sav', 'rb') as f:
        model = pickle.load(f)

    context = {
        'symptoms': Features.objects.all()
        }
    return render(request, "main/home.html", context)
