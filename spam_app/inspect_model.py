import pickle
import os
import sys

try:
<<<<<<< HEAD
    # Look in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'svm_model.pkl')
    with open(model_path, 'rb') as f:
=======
    with open('svm_model.pkl', 'rb') as f:
>>>>>>> 50aa913d08abbd462083c549b0a25b36d6e6843b
        model = pickle.load(f)
    print(f"Model Classes: {model.classes_}")
    print(f"Type of classes: {type(model.classes_[0])}")
except Exception as e:
    print(f"Error loading model: {e}")
