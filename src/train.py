import airfrans

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils import formatize_training_data
from data_5_digits import makecamberline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data_folder='/home/brice/Documents/Airfrans_ml/data/Dataset'
# Charger le dataset AirfRANS
if __name__=="__main__":
    rough_data=airfrans.dataset.load(data_folder, 'scarce', train=True)
    df_airfrans_data=formatize_training_data(rough_data)[1]
    
X= df_airfrans_data[['Re', 'Mach', 'AoA', 'Camber(NACA)', 'Pos max Camber (NACA)', 'Profile Type', 'Thickness']]
y= df_airfrans_data['Cd']
# Séparer les données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
dd
# Définir une expérience MLflow
mlflow.set_experiment("AirfRANS-ML")

with mlflow.start_run():
    # Entraîner le modèle
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Faire des prédictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Enregistrer les paramètres et métriques
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)

    # Sauvegarder le modèle
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Modèle entraîné avec MSE: {mse}")
