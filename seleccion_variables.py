"""
@author: Andres Fernando Guaca
"""

import  pandas as pd
import h2o        

def tipificar_h2o(train):
    """
    Esta función lee un objeto data frame de h2o y convierte las columnas con menos 
    de 20 categorías en factores.
    """    
    A=train.as_data_frame()
    for col in train.col_names:
        if pd.crosstab(A[col],columns="frecuencia").shape[0]<=20:
           train[col]= train[col].asfactor()    

def seleccion_variables(modelo,datos,by_select="T_PRO",num_var=30,s=1234,var_total=[],var_select=[]):
    """
    Esta funció realiza una selección de variables donde los parametros son:   
    modelo: un modelo h2o con el que se va a realizar la selección de variables.
    datos: lee un objeto data_coin.    
    by_select: el indicador por el que se va a tener en cuenta para seleccionar la mejor variable. En este caso, tasa de pronóstico datos training PRO o 
    tasa de pronóstico de datos test T_PRO.     
    num_var: número de máximo de variables.    
    s: semilla de partición de datos training-test.    
    var_total: total  de variables a analizar.    
    var_select: variables incluidas inicialmente.
    """            
    pasos=[]
    P2=0
    var_total=list(set(var_total) - set(var_select))
    train = h2o.H2OFrame(datos[0])
    tipificar_h2o(train)
    splits = train.split_frame(ratios=[0.7], seed=s)   
    for k in range(num_var):
        ERROR=[]
        VAR=[]
        PRO=[]
        T_PRO=[]
        T_PRO_BUY=[]
        PRO_BUY=[]

        for i in var_total:
            try:            
                mod = modelo
                mod.train(x=list(set(var_select) | set([i])), y="Tendencia", training_frame=splits[0])
                confusion_train=mod.confusion_matrix().table.as_data_frame()
                confusion_test=mod.model_performance(splits[1]).confusion_matrix().table.as_data_frame()                     
                
                PRO.append((confusion_train.iloc[0,1]+confusion_train.iloc[1,2])/(confusion_train.iloc[0,2]+confusion_train.iloc[1,1]+confusion_train.iloc[0,1]+confusion_train.iloc[1,2]))
                T_PRO.append((confusion_test.iloc[0,1]+confusion_test.iloc[1,2])/(confusion_test.iloc[0,2]+confusion_test.iloc[1,1]+confusion_test.iloc[0,1]+confusion_test.iloc[1,2]))
                T_PRO_BUY.append(confusion_test.iloc[1,2]/(confusion_test.iloc[1,1]+confusion_test.iloc[1,2]))                    
                PRO_BUY.append(confusion_train.iloc[1,2]/(confusion_train.iloc[1,1]+confusion_train.iloc[1,2]))                    
                VAR.append(i)           

            except:
                ERROR.append(i)
                print("Error con la variable: "+i)
                
        var_total=list(set(var_total) - set(ERROR))        
        A=pd.DataFrame({"VAR":VAR,"PRO":PRO,"T_PRO":T_PRO,"PRO_BUY":PRO_BUY,"T_PRO_BUY":T_PRO_BUY})        
        B=A.sort_values(by_select,ascending=False)        
        pasos.append(B)                       
        var_total=list(set(var_total)-set([B.VAR.iloc[0]]))    
        if (B[by_select].iloc[0]-P2)>0.001:
            P2=B[by_select].iloc[0]
            var_select.append(B.VAR.iloc[0])
        else:
            break        
        print(var_select)
          
    return var_select, pasos
