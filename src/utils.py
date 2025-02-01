import pyvista as pv
from data_5_digits import makecamberline
import airfrans
import pandas as pd
import os 
data_folder='/home/brice/Documents/Airfrans_ml/data/Dataset'
if __name__=="__main__":
    data=airfrans.dataset.load(data_folder, 'scarce', train=True)
def formatize_training_data(data):
    """
    This function takes the data from the dataset and formatize it to be used in the training.  
    In: rough data from aifrans dataset
    Out: Dataframe with the following columns: Simulations, Re, Mach, AoA, Camber(NACA), Pos max Camber (NACA), Profile Type, Thickness, Cl, Cd, Fitness
    First Output is a dictionary with the simulation name as key and the value is a list of the following values: Re, Mach, AoA, Camber(NACA), Profile Type, Thickness, Pos max Camber (NACA), Cl, Cd, Fitness
    Second output is a dataframe with the following columns: Simulations, Re, Mach, AoA, Camber(NACA), Pos max Camber (NACA), Profile Type, Thickness, Cl, Cd, Fitness
    """
    list_simulations_name=data[-1]
    aero_load_simulations_df = {
    
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
        }
    
    # Dictionaire Simulation Re, Mach, AoA, thickness, camber lift, drag,, fitness
    
    aero_load_simulations={}
    for i in range(len(list_simulations_name)):
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
    
    return aero_load_simulations, pd.DataFrame(aero_load_simulations_df)
