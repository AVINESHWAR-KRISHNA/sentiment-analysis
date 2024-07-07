import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'training_data.csv')

PREDICTION_PATH = os.path.join(BASE_DIR, 'datasets', 'predict_dataset.csv')

MODELS_DIR = os.path.join(BASE_DIR, 'models')

OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

TRAIN_MODELS = True  # Set to False if you don't want to retrain the models