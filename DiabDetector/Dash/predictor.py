import numpy as np
from Dash.svm import SVM
from sklearn.preprocessing import StandardScaler
import os
import pickle


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # If the pickled object's module is __main__, remap it to Dash.svm
        if module == "__main__":
            module = "Dash.svm"
        return super().find_class(module, name)



base_dir = os.path.dirname(os.path.abspath(__file__))


svm_path = os.path.join(base_dir, 'svm.pkl')


with open(svm_path, 'rb') as f:
    model = CustomUnpickler(f).load()


scaler_path = os.path.join(base_dir, 'scaler.pkl')


with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)


modelsLoaded = True

def predictIt(input_vector):
    input_vector = np.array(input_vector).reshape(1, -1)
    input_vector = scaler.transform(input_vector)
    result = np.sign(model.predict(input_vector))
    return "You might have diabetes." if result == 1 else "No Diabetes"