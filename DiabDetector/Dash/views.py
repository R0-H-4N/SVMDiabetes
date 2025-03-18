from django.shortcuts import render
from .predictor import predictIt

def home(request):
    return render(request, 'home.html')

def predict(request):
    if request.method == 'POST':
        input_vector = []
        input_vector.append(float(request.POST['Pregnancies']))
        input_vector.append(float(request.POST['Glucose']))
        input_vector.append(float(request.POST['BloodPressure']))
        input_vector.append(float(request.POST['Weight']) / (float(request.POST['Height']) / 100) ** 2)
        input_vector.append(float(request.POST['Age']))
        result = predictIt(input_vector)
        return render(request, 'predict.html', {'prediction': result})
    else:
        return render(request, 'predict.html')
