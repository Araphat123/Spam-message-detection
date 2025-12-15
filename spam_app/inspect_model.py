import pickle
import os
import sys

try:
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"Model Classes: {model.classes_}")
    print(f"Type of classes: {type(model.classes_[0])}")
except Exception as e:
    print(f"Error loading model: {e}")
