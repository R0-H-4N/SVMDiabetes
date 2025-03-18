from svm import *
from sklearn.preprocessing import StandardScaler
import pickle

with open('svm.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict(input_vector):
    input_vector = np.array(input_vector).reshape(1, -1)
    input_vector = scaler.transform(input_vector)
    result = np.sign(model.predict(input_vector))
    return "Diabetes" if result == 1 else "No Diabetes"


healthy_vector = [1, 85, 70, 22, 25]
very_unhealthy_vector = [10, 200, 100, 35, 60]
print(predict(healthy_vector))
print(predict(very_unhealthy_vector))