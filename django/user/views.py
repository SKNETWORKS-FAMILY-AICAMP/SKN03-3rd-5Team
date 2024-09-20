from django.shortcuts import render

# Create your views here.

def go_homepages(request):

    return render(request, "user/index.html")