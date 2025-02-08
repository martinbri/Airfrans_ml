import airfrans
from tqdm import tqdm
import matplotlib
import pyvista as pv
import pickle
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
from sklearn.model_selection import train_test_split
from data_5_digits import makecamberline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score,mean_absolute_error, mean_squared_error, r2_score
import pytest
#import sklearn
from matplotlib.path import Path
import os 
data_folder='/home/brice/Documents/Airfrans_ml/data/Dataset'
if __name__=="__main__":
    pass
    #data=airfrans.dataset.load(data_folder, 'scarce', train=True)
def make_fields_data_base ():
    """
    This function plots the fields of the airfoil
    """
    #data=airfrans.dataset.load(data_folder, 'scarce', train=True)
    list_simu_name=os.listdir(data_folder)    
    aero_load_simulations_df = {

        "Simulations": [],
        "Pressure": [],
       
        "Cl": [],
        "Cd": [],
        "Fitness": []
        }
    
    for i in tqdm(range(len(list_simu_name)),desc="Formating data",ncols=100,unit='it'):
        if os.path.splitext(list_simu_name[i])[1].lower() == '.json':
            # Proceed with further actions if the file is a JSON file
            print(f"The file {list_simu_name[i]} is a JSON file.")
            continue
        simu=airfrans.Simulation(data_folder,list_simu_name[i])
        (cd,_,_),(cl,_,_)=simu.force_coefficient()

        coordinates=simu.position
        pressure=simu.pressure
        
        
        
        # make the pressure fields
        airfoil_coordinate=simu.airfoil_position
        plt.plot(simu.airfoil_position[:,0],simu.airfoil_position[:,1])
        plt.plot(simu.airfoil_position[:,0],simu.airfoil_position[:,1],'go')
        plt.plot(simu.airfoil_position[0,0],simu.airfoil_position[0,1],'ro')
        plt.ylim(-0.33,0.33)
        plt.xlim(-0.5,1.5)
        _,_,_,profile=makecamberline(simu)

        plt.plot(profile[:,0],profile[:,1],'k-')
        #plt.plot(profile[:,0],profile[:,1],'go')
        #plt.plot(profile[0,0],profile[0,1],'ro')
        plt.close()
        

    
        poligon=Path(profile,closed=True)


    

        

    
        grid_x, grid_y = np.mgrid[-0.5:1.5:1000j, -0.33:0.33:333j]
        points=np.c_[grid_x.reshape(-1,1),grid_y.reshape(-1,1)]
    
        inside=poligon.contains_points(points)
      
        inside=inside.reshape(1000,333)
    
  
    

        # Interpoler les données sur la grille
        grid_pressure = griddata(coordinates, pressure, (grid_x, grid_y), method='linear', fill_value=0)
        grid_pressure[inside]=-0.000
        maskedgridpressure = np.ma.masked_where(grid_pressure == 0.000, grid_pressure)
        cmap=matplotlib.cm.viridis
        cmap.set_bad(color='grey')
   
        # Create filled contour plot
        plt.figure(figsize=(10, 8))

        # Create imshow plot
        plt.imshow(maskedgridpressure[:,:,0].T,interpolation='bilinear',origin='lower',cmap=cmap)
    
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('colorbar_label')
    
        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Pressurefield')    
        plt.close()
        aero_load_simulations_df["Simulations"].append(list_simu_name[i])
        aero_load_simulations_df["Pressure"].append(grid_pressure)
        aero_load_simulations_df["Cl"].append(cl)
        aero_load_simulations_df["Cd"].append(cd)
        aero_load_simulations_df["Fitness"].append(cl/cd)
        
        

    with open('data_pressure_field.pkl', 'wb') as f:
            pickle.dump(aero_load_simulations_df, f)
    print("Data has been cached.")
    
    return pd.DataFrame(aero_load_simulations_df)

make_fields_data_base()
plt.show()
dd

def formatize_training_data(data,aero_load_simulations_df = {

        "Simulations": [],
        "Re": [],
        "Mach": [],
        "AoA": [],
        "Camber(NACA)": [],
        "Pos max Camber (NACA)": [],
        "Profile Type": [],
        "Thickness": [],
        "Cl": [],
        "Cd": [],
        "Fitness": []
        }):
    """
    This function takes the data from the dataset and formatize it to be used in the training.  
    In: rough data from aifrans dataset
    Out: Dataframe with the following columns: Simulations, Re, Mach, AoA, Camber(NACA), Pos max Camber (NACA), Profile Type, Thickness, Cl, Cd, Fitness
     output is a dataframe with the following columns: Simulations, Re, Mach, AoA, Camber(NACA), Pos max Camber (NACA), Profile Type, Thickness, Cl, Cd, Fitness
    """
    list_simulations_name=data[-1]
    
    
    # Dictionaire Simulation Re, Mach, AoA, thickness, camber lift, drag,, fitness
    
    aero_load_simulations={}
    for i in tqdm(range(len(list_simulations_name)),desc="Formating data",ncols=100,unit='it'):
        simu=airfrans.Simulation(data_folder,list_simulations_name[i])
        (cd,_,_),(cl,_,_)=simu.force_coefficient()
        AoA=simu.angle_of_attack
        U_infty=simu.inlet_velocity
        Re=U_infty*1/simu.NU
        Mach=U_infty/simu.C
        

        words_simulation_name=list_simulations_name[i].split('_')
        if len(words_simulation_name)==7:
            thickness=float(words_simulation_name[-1])
            max_camber=float(words_simulation_name[-3])
            max_camber_pos=float(words_simulation_name[-2])
            is_double_profile=0
        elif len(words_simulation_name)==8:
            thickness=float(words_simulation_name[-1])
            optimal_Cl=float(words_simulation_name[-4])
            max_camber_pos=float(words_simulation_name[-3])/2 ##*2 is to make it co,nsistant with the 4 digits
        
            max_camber=makecamberline(simu)[0]
            is_double_profile=float(words_simulation_name[-2])+1
        else:
            raise ValueError('The simulation name is not well formatted')

        aero_load_simulations[list_simulations_name[i]]=[Re,AoA,Mach,max_camber,is_double_profile,thickness,max_camber_pos,cd,cl,cl/cd]
        aero_load_simulations_df["Simulations"].append(list_simulations_name[i])
        aero_load_simulations_df["Re"].append(Re)
        aero_load_simulations_df["Mach"].append(Mach)       
        aero_load_simulations_df["AoA"].append(AoA)
        aero_load_simulations_df["Camber(NACA)"].append(max_camber) 
        aero_load_simulations_df["Pos max Camber (NACA)"].append(max_camber_pos)
        aero_load_simulations_df["Profile Type"].append(is_double_profile)
        aero_load_simulations_df["Thickness"].append(thickness)
        aero_load_simulations_df["Cl"].append(cl)
        aero_load_simulations_df["Cd"].append(cd)
        aero_load_simulations_df["Fitness"].append(cl/cd)
    
    return pd.DataFrame(aero_load_simulations_df)


class ML_pipeline:
    def __init__(self,data_set,model,features_input,features_output,need_formatize=False,formatizer=None,Ml_flow=False,params={},is_classification=True,nclass=10):
        """
        This class is a pipeline to train a model on the dataset. It takes argument the dataset and the sklearn model.
        
        IN: data_set: pandas DataFrame, model: sklearn model formatizer: function to formatize the data, need_formatize: boolean to indicate formatization is required.
        OUT: None


        """
        assert isinstance(data_set,pd.DataFrame),"{} doit être un DataFrame pandas".format(data_set)
        if need_formatize:
            assert formatizer is not None, "You need to provide a formatizer function"
            self.data_set=formatizer(data_set)
        else: 
            self.data_set=data_set
        self.X=self.data_set[features_input]
        self.y=self.data_set[features_output]
        if is_classification:
            verbose=False

            self.y = self._label_data(nclass,verbose)
        
        self.is_classification=is_classification

        self.model=model(**params)
        self.Ml_flow=Ml_flow

        self.model_type=model
        self.features_input=features_input
        self.features_output=features_output

        if self.Ml_flow:
            mlflow.start_run()
            mlflow.log_param('features_input',features_input)
            mlflow.log_param('features_output',features_output)
            mlflow.log_param('model',model)
            mlflow.log_param('data_set',data_set)

    def _label_data(self,nclass,verbose=False):
        """
        This function labels the data into nclass classes
        """
        y=self.y
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
        return y

        

    def _split_data(self,train_size=0.8):
        """
        This function splits the data into train and test set.
        """
        self.X_train,self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1-train_size, random_state=42)


    def _train(self):
        """
        This function trains the model on the dataset
        """
        self.model.fit(self.X_train,self.y_train)
        if hasattr(self.model,'coef_'):
            self.weights=self.model.coef_
            self.intercept=self.model.intercept_
        if self.is_classification:

            accuracy = accuracy_score(self.y_train, self.model.predict(self.X_train),average='weighted')
            precision=precision_score(self.y_train, self.model.predict(self.X_train),average='weighted')
            recall=recall_score(self.y_train, self.model.predict(self.X_train),average='weighted')
            f1=f1_score(self.y_train, self.model.predict(self.X_train),average='weighted')
        else :
            score = r2_score(self.y_train, self.model.predict(self.X_train))
            abs_error=mean_absolute_error(self.y_train, self.model.predict(self.X_train))
            mse=mean_squared_error(self.y_train, self.model.predict(self.X_train))
            f1=None
            print('score from training: {}, abs_error {}, mse: {}'.format(score,abs_error,mse))    
    
        if self.Ml_flow:
            mlflow.log_param('weights',self.weights)
            mlflow.log_param('model',self.model)


           # Loguer les paramètres, les métriques et le modèle avec MLflow
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_metric("accuracy training", accuracy)
            mlflow.log_metric("precision training", precision)
            mlflow.log_metric("recall training", recall)
            mlflow.log_metric("f1 training", f1)
        return False

    def _test(self):
        """
        This function tests the model on the dataset
        """
        y_pred=self.model.predict(self.X_test)
        if self.is_classification:
            accuracy = accuracy_score(self.y_test, y_pred,average='weighted')
            precision=precision_score(self.y_test, y_pred,average='weighted')
            recall=recall_score(self.y_test, y_pred,average='weighted')
            f1=f1_score(self.y_test, y_pred,average='weighted')
        else :
            score = r2_score(self.y_test, y_pred)
            abs_error=mean_absolute_error(self.y_test, y_pred)
            mse=mean_squared_error(self.y_test, y_pred)
            f1=None
            print('score from testing: {}, abs_error {}, mse: {}'.format(score,abs_error,mse))   
        if self.Ml_flow:
           # Loguer les paramètres, les métriques et le modèle avec MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)
        return False
    def _cm_matrix(self):
        """
        This function returns the confusion matrix
        """
           #Save the input/output quantities
    # Dictionnaire contenant les colonnes d'entrée et de sortie
        column_info = {
            "input_columns": list(self.X_train.columns),
            "output_column": self.y_train.name if isinstance(self.y_train, pd.Series) else list(self.y_train.columns)}

    # Logger dans MLflow
        mlflow.log_dict(column_info, "column_labels.json")
    

    # Save the model to MLflow with signature
        mlflow.set_tag("mlflow.note.content", "{} to classify {} depending on {} in {} classes.".format(self.model_type,self.features_input, self.features_output,nclass))
 
    
        signature = infer_signature(np.array(X_test), model.predict(X_test) )
        mlflow.sklearn.log_model(model, artifact_path="logistic_regression_model", signature=signature)

        # Log additional training info if provided
        if training_info is not None:
            mlflow.set_tag("Training info", training_info)

    # End the MLflow run

   

    # Compute and plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
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
        return confusion_matrix(self.y_test, self.model.predict(self.X_test))
    
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
        df_airfrans_data=formatize_training_data(rough_data)
        with open(cache_file, 'wb') as f:
            pickle.dump(df_airfrans_data, f)
        print("Data has been cached.")
features_in=['Re', 'Mach', 'AoA', 'Camber(NACA)', 'Pos max Camber (NACA)', 'Profile Type', 'Thickness']

#features_in=['Re', 'AoA', 'Camber(NACA)', 'Pos max Camber (NACA)', 'Thickness']
from sklearn.ensemble import RandomForestRegressor
params={'n_estimators':1000, 'max_depth':10,'random_state':42}

linear_regression=ML_pipeline(df_airfrans_data, RandomForestRegressor, features_input=features_in, features_output='Fitness', need_formatize=False, formatizer=None, Ml_flow=False, params=params, is_classification=False, nclass=10)




#linear_regression=ML_pipeline(df_airfrans_data, sklearn.linear_model.LinearRegression, features_in, 'Fitness', need_formatize=False, formatizer=None, Ml_flow=False, params={}, is_classification=False, nclass=10)
linear_regression._split_data()
linear_regression._train()
linear_regression._test()

plt.plot(df_airfrans_data['AoA'],df_airfrans_data['Fitness'],'go')
plt.plot(df_airfrans_data['AoA'],linear_regression.model.predict(linear_regression.X),'bo')

plt.show()

y_pred=linear_regression.model.predict(linear_regression.X_test)
# Résidus
residuals = (linear_regression.y_test - y_pred)/linear_regression.y_test 

from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(linear_regression.model, linear_regression.X_train, linear_regression.y_train, cv=5, scoring='neg_mean_squared_error')

# Moyenne des scores pour les ensembles d'entraînement et de test
train_errors = -np.mean(train_scores, axis=1)
test_errors = -np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_errors, label="Erreur d'entraînement")
plt.plot(train_sizes, test_errors, label="Erreur de validation")
plt.xlabel('Nombre d’exemples d’entraînement')
plt.ylabel('Erreur MSE')
plt.title('Courbe d’apprentissage')
plt.legend()
plt.show()


# Visualisation des résidus
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.title('Graphique des résidus')
plt.show()
LinearRegression()
#logistic_regression = ML_pipeline(df_airfrans_data, sklearn.linear_model.LogisticRegression, ['Re', 'Mach', 'AoA', 'Camber(NACA)', 'Pos max Camber (NACA)', 'Profile Type', 'Thickness'], 'Fitness', need_formatize=False, formatizer=None, Ml_flow=False, params={'verbose':True}, is_classification=True, nclass=10)