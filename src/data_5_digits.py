import pandas as pd
import matplotlib.pyplot as plt
import airfrans
def make_5_digits_camber():
# Define the data
    data = {
        "Description": [
            "5% standard", "10% standard", "15% standard", "20% standard", "25% standard",
            "10% reflex", "15% reflex", "20% reflex", "25% reflex"
        ],
        "Digits": [10, 20, 30, 40, 50, 21, 31, 41, 51],
        "Camber Position (%)": [5, 10, 15, 20, 25, 10, 15, 20, 25],
        "r": [0.0580, 0.1260, 0.2025, 0.2900, 0.3910, 0.1300, 0.2170, 0.3180, 0.4410],
        "k1": [361.400, 51.640, 15.957, 6.643, 3.230, 51.990, 15.793, 6.520, 3.191],
        "k2/k1": [None, None, None, None, None, 0.000764, 0.00677, 0.0303, 0.1355]
        }

    df = pd.DataFrame(data)
    plt.plot(df["Camber Position (%)"], df["k2/k1"],'bo')
    plt.show()
import numpy as np
def makecamberline(simu):
    """
    This function returns the camber line of a NACA profile
    In: name of simulation
    Out: Camber (Naca notation), equation of the camber line X,Y
    """
    list_simu_name=simu.name.split('_')
    if len(list_simu_name)==7:
        thickness=list_simu_name[-1]
        camber=list_simu_name[-3]
        max_camber_pos=list_simu_name[-2]
        is_4_digit=True
        params=(camber,max_camber_pos)
    elif len(list_simu_name)==8:
        thickness=float(list_simu_name[-1])
        optimal_Cl=float(list_simu_name[-4])
        max_camber_pos=float(list_simu_name[-3])
        is_4_digit=False
        is_double_profile=float(list_simu_name[-2])
        params=(optimal_Cl,max_camber_pos,is_double_profile)
    else:
        raise ValueError('The simulation name is not well formatted')
    X=np.linspace(0,1,400)
    camber_line=airfrans.naca_generator.camber_line(params,X)
    camber_Naca=np.max(camber_line[0])*100#Chord=1 resultat au format Naca
    return camber_Naca,X,camber_line[0]
    


if __name__=="__main__":
    make_5_digits_camber()