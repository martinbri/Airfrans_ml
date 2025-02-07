import airfrans
import pickle
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from utils import formatize_training_data
from data_5_digits import makecamberline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score



data_folder='/home/brice/Documents/Airfrans_ml/data/Dataset'
# Charger le dataset AirfRANS
if __name__=="__main__":
    # Define the path where you will save the cached data
    cache_file = 'data_cache.pkl'
    """Load the data from cache or load it afresh and save to cache."""
    if os.path.exists(cache_file):
        # If the cached file exists, load the data from there
        print("Loading data from cache...")
        with open(cache_file, 'rb') as f:
            df_airfrans_data = pickle.load(f)
    else:
        # If the cached file does not exist, load the dataset and save it to the cache
        rough_data=airfrans.dataset.load(data_folder, 'scarce', train=True)
        df_airfrans_data=formatize_training_data(rough_data)[1]
        with open(cache_file, 'wb') as f:
            pickle.dump(df_airfrans_data, f)
        print("Data has been cached.")
    

# Préparer les données    

def logistic_regression(features,target,verbose=False,nclass=10,plot_conf_matrix=True,training_info=None,regularization=None,experiment_name="Airfrans_ml"):
    #X= df_airfrans_data[['Re', 'Mach', 'AoA', 'Camber(NACA)', 'Pos max Camber (NACA)', 'Profile Type', 'Thickness']]
    X= df_airfrans_data[features]#,'Camber(NACA)', 'Pos max Camber (NACA)',"Thickness"]]

    y= df_airfrans_data[target]

    if verbose:
        plt.hist(y, bins=30, density=True, alpha=0.5, label="Histogram")
        plt.show()
        plt.plot(y,'go')
        plt.show()
        # Sép   arer les données en train/test
    listed_y = np.sort(y)
    back_up_y=y
    nclass=10
    n_data=len(np.sort(y))
    for i in range(nclass):
        classifier=(back_up_y>=listed_y[int(n_data/nclass*i)])&(back_up_y<=listed_y[int(n_data/nclass*(i+1)-1)])
        y[classifier] = i
        if verbose:
            plt.plot(y,'go')
            plt.show()




    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Définir une expérience MLflow
    print('ded')
    #mlflow.set_tracking_uri("http://localhost:5000")
    
    
    mlflow.start_run()
    mlflow.set_experiment(experiment_name)
    
    mlflow.set_tag("dataset", "Airfrans")
    print("Starting run")


    # Entraîner un modèle de régression logistique
    params={'multi_class':'multinomial', 'solver':'lbfgs', 'max_iter':1000,'verbose':0,'penalty':regularization}

    model =LogisticRegression(**params)
    model.fit(X_train, y_train)


    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluer les performances du modèle (précision ici)
    accuracy = accuracy_score(y_test, y_pred,average='weighted')
    precision=precision_score(y_test, y_pred,average='weighted')
    recall=recall_score(y_test, y_pred,average='weighted')
    f1=f1_score(y_test, y_pred,average='weighted')
    

    # Loguer les paramètres, les métriques et le modèle avec MLflow
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    

    # Sauvegarder le modèle dans MLflow

    mlflow.sklearn.log_model(sk_model=model, artifact_path="logistic_regression_model",signature=infer_signature(X_train, model.predict(X_train)))
    mlflow.log_params(params)
    # Fin de l'exécution MLflow
    if training_info is not None:
        mlflow.set_tag("Training info",training_info)
    mlflow.end_run()

    print(f"Accuracy: {accuracy}")
    print(f"model's weight: {model.coef_}")
    print(f"model's bias: {model.intercept_}")
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix using Seaborn heatmap
    if plot_conf_matrix:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["0", "1","2", "3"], yticklabels=["0", "1","2", "3"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    return accuracy

def standard_vector_machines(features,target,verbose=False,nclass=10,plot_conf_matrix=True,regularization=None):
    #X= df_airfrans_data[['Re', 'Mach', 'AoA', 'Camber(NACA)', 'Pos max Camber (NACA)', 'Profile Type', 'Thickness']]
    X= df_airfrans_data[features]#,'Camber(NACA)', 'Pos max Camber (NACA)',"Thickness"]]

    y= df_airfrans_data[target]

    if verbose:
        plt.hist(y, bins=30, density=True, alpha=0.5, label="Histogram")
        plt.show()
        plt.plot(y,'go')
        plt.show()
        # Sép   arer les données en train/test
    listed_y = np.sort(y)
    back_up_y=y
    nclass=10
    n_data=len(np.sort(y))
    for i in range(nclass):
        classifier=(back_up_y>=listed_y[int(n_data/nclass*i)])&(back_up_y<=listed_y[int(n_data/nclass*(i+1)-1)])
        y[classifier] = i
        if verbose:
            plt.plot(y,'go')
            plt.show()

    ff


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #print(X_train)
    ff
        

    # Définir une expérience MLflow
    mlflow.start_run()
    print("Starting run")

    # Entraîner un modèle de régression logistique
    model =LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000,verbose=0,penalty=regularization)
    model.fit(X_train, y_train)
    signature=infer_signature(X_train, model.predict(X_train))

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Évaluer les performances du modèle (précision ici)
    accuracy = accuracy_score(y_test, y_pred)

    # Loguer les paramètres, les métriques et le modèle avec MLflow
    mlflow.log_param("model_type", "Logistic Regression")
    mlflow.log_metric("accuracy", accuracy)

    # Sauvegarder le modèle dans MLflow
    mlflow.sklearn.log_model(model, "logistic_regression_model")

    # Fin de l'exécution MLflow
    mlflow.end_run()

    print(f"Accuracy: {accuracy}")
    print(f"model's weight: {model.coef_}")
    print(f"model's bias: {model.intercept_}")
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix using Seaborn heatmap
    if plot_conf_matrix:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["0", "1","2", "3"], yticklabels=["0", "1","2", "3"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    return accuracy

def logistic_regression_chat(features, target, verbose=False, nclass=10, plot_conf_matrix=True, training_info=None, regularization=None, experiment_name="Airfrans_ml",ml_flow=False):
    X = df_airfrans_data[features]
    y = df_airfrans_data[target]
    

    if verbose:
        plt.hist(y, bins=30, density=True, alpha=0.5, label="Histogram")
        plt.show()
        plt.plot(y, 'go')
        plt.show()

    listed_y = np.sort(y)
    back_up_y = y
    n_data = len(np.sort(y))
    for i in range(nclass):
        classifier = (back_up_y >= listed_y[int(n_data/nclass*i)]) & (back_up_y <= listed_y[int(n_data/nclass*(i+1)-1)])
        y[classifier] = i
        if verbose:
            plt.plot(y, 'go')
            plt.show()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Define a MLflow experiment
    #mlflow.set_tracking_uri("http://localhost:5000")

    if ml_flow:
    
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.set_tag("dataset", "Airfrans")

    # Define model parameters
    params = {'multi_class': 'multinomial', 'solver': 'lbfgs', 'max_iter': 1000, 'verbose': 0, 'penalty': regularization}

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_training_set = accuracy_score(y_train, model.predict(X_train))

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log parameters, metrics, and model to MLflow
    if ml_flow:
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

    #Save the input/output quantities
    # Dictionnaire contenant les colonnes d'entrée et de sortie
        column_info = {
            "input_columns": list(X_train.columns),
            "output_column": y_train.name if isinstance(y_train, pd.Series) else list(y_train.columns)}

    # Logger dans MLflow
        mlflow.log_dict(column_info, "column_labels.json")
    

    # Save the model to MLflow with signature
        mlflow.set_tag("mlflow.note.content", "Logistic Regression to classify {} depending on {} in {} classes.".format(target, features,nclass))
 
    
        signature = infer_signature(np.array(X_test), model.predict(X_test) )
        mlflow.sklearn.log_model(model, artifact_path="logistic_regression_model", signature=signature)

        # Log additional training info if provided
        if training_info is not None:
            mlflow.set_tag("Training info", training_info)

    # End the MLflow run

    print(f"Accuracy: {accuracy}")
    print(f"Accuracy trainng: {accuracy_training_set} ")

    print(f"Model's weights: {model.coef_}")
    print(f"Model's bias: {model.intercept_}")
   

    # Compute and plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if True:
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('conf_matrix.png')
        if plot_conf_matrix:
            plt.show()
        if ml_flow:
            mlflow.log_artifact('conf_matrix.png')
    
            os.remove('conf_matrix.png')
            mlflow.end_run()

    


    # Select only the first two features for 2D visualization
    if X.shape[1] > 2:
        print("Warning: Only the first two features are used for visualization.")
        X = X.iloc[:, :2].values  

    else:
        # Create a mesh grid
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                         np.linspace(y_min, y_max, 100))

        # Predict for every point in the grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.figure(figsize=(8, 6))

        X_ = df_airfrans_data[features]
        y_ = df_airfrans_data[target]
        print(y)
        print(type(y))
    

        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title("Decision Boundary of Logistic Regression")
        plt.show()

# Call function with trained model

    return accuracy

logistic_regression_chat(features=['AoA', 'Pos max Camber (NACA)', 'Thickness'],target='Cd',verbose=False,nclass=5,plot_conf_matrix=True,regularization=None)