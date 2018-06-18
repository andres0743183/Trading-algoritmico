

import h2o
from h2o.grid.grid_search import H2OGridSearch
#import math
import pandas as pd
#import datos as dt
#import os.path as path
import Mercado_opt
#import matplotlib.pyplot as plt
#import funciones

def procesamiento_resultados_binario(gs,splits,datos):    
        resultados=gs.sorted_metric_table()
        tem1=pd.DataFrame({"T_SS":i.model_performance(splits[1]).confusion_matrix().table.as_data_frame().iloc[0,1],
              "T_SB":i.model_performance(splits[1]).confusion_matrix().table.as_data_frame().iloc[0,2],
               "T_BS":i.model_performance(splits[1]).confusion_matrix().table.as_data_frame().iloc[1,1],
               "T_BB":i.model_performance(splits[1]).confusion_matrix().table.as_data_frame().iloc[1,2]} for i in gs)
        tem1["T_PRO"]=(tem1["T_BB"]+tem1["T_SS"])/(tem1["T_BB"]+tem1["T_SS"]+tem1["T_BS"]+tem1["T_SB"])
        
        tem2=pd.DataFrame({"SS":i.confusion_matrix().table.as_data_frame().iloc[0,1],
                      "SB":i.confusion_matrix().table.as_data_frame().iloc[0,2],
                       "BS":i.confusion_matrix().table.as_data_frame().iloc[1,1],
                       "BB":i.confusion_matrix().table.as_data_frame().iloc[1,2]} for i in gs)

        tem2["PRO"]=(tem2["BB"]+tem2["SS"])/(tem2["BB"]+tem2["SS"]+tem2["BS"]+tem2["SB"])        
        resultados1=pd.concat([resultados,tem1,tem2],axis=1)         
        GN=[]
        TRAN=[]
        MEDIANA=[]
        test = h2o.H2OFrame(datos[1]) 
        for mod in gs:        
            prediccion=mod.predict(test).as_data_frame()
            precios=datos[1][["date","low","close"]].reset_index()
            analisis=Mercado_opt.mercado_simulacion(precios,prediccion["predict"],stop=0.1,Invercion=1,ct=0.002)
            #plt.figure()  
            GN.append(analisis["GN"].cumsum().iloc[-1,])
            TRAN.append(analisis.shape[0])
            MEDIANA.append(analisis["GN"].median())
                    
        return resultados1


def tipificar_h2o(train):
    A=train.as_data_frame()
    for col in train.col_names:
        if pd.crosstab(A[col],columns="frecuencia").shape[0]<=20:
           train[col]= train[col].asfactor()    

def validacion_r(modelo, hyper_parameters, datos, variables, semilla=1234):    
    h2o.init(max_mem_size=14) 
    train = h2o.H2OFrame(datos[0])
    tipificar_h2o(train)
    splits = train.split_frame(ratios=[0.7], seed=semilla)
    gs = H2OGridSearch(modelo, hyper_params=hyper_parameters)
    gs.train(x=variables, y="Tendencia", training_frame=splits[0])
    resultados=procesamiento_resultados_binario(gs,splits,datos)
    h2o.remove_all()
    return(resultados)
 

