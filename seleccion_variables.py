#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 01:12:11 2018

@author: andres
"""

#import os.path as path
import  pandas as pd

import Mercado_opt 
import h2o        

def tipificar_h2o(train):
    A=train.as_data_frame()
    for col in train.col_names:
        if pd.crosstab(A[col],columns="frecuencia").shape[0]<=20:
           train[col]= train[col].asfactor()    


def seleccion_variables(modelo,datos,by_select="T_PRO",num_var=10,tran_min=20,s=1234,var_total=[],var_select=[]): 
    #h2o.init(max_mem_size=14)  
    pasos=[]
    P2=0
    var_total=list(set(var_total) - set(var_select))
    train = h2o.H2OFrame(datos[0])
    test = h2o.H2OFrame(datos[1])
    tipificar_h2o(train)
    splits = train.split_frame(ratios=[0.7], seed=s)
   
    for k in range(num_var):
        ERROR=[]
        VAR=[]
        PRO=[]
        T_PRO=[]
        T_PRO_BUY=[]
        PRO_BUY=[]
        GN=[]
        TRAN=[]
        MEDIANA=[]
        #i=  "volume_9"
        for i in var_total:
            try:            
                mod = modelo
                mod.train(x=list(set(var_select) | set([i])), y="Tendencia", training_frame=splits[0])
                confusion_train=mod.confusion_matrix().table.as_data_frame()
                confusion_test=mod.model_performance(splits[1]).confusion_matrix().table.as_data_frame()                     
                prediccion=mod.predict(test).as_data_frame()
                precios=datos[1][["date","low","close"]].reset_index()
                analisis=Mercado_opt.mercado_simulacion(precios,prediccion["predict"],stop=0.1,Invercion=1,ct=0.002)
                if analisis.shape[0]!=0:
                    GN.append(analisis["GN"].cumsum().iloc[-1,])
                    TRAN.append(analisis.shape[0])
                    MEDIANA.append(analisis["GN"].median())
                    PRO.append((confusion_train.iloc[0,1]+confusion_train.iloc[1,2])/(confusion_train.iloc[0,2]+confusion_train.iloc[1,1]+confusion_train.iloc[0,1]+confusion_train.iloc[1,2]))
                    T_PRO.append((confusion_test.iloc[0,1]+confusion_test.iloc[1,2])/(confusion_test.iloc[0,2]+confusion_test.iloc[1,1]+confusion_test.iloc[0,1]+confusion_test.iloc[1,2]))
                    T_PRO_BUY.append(confusion_test.iloc[1,2]/(confusion_test.iloc[1,1]+confusion_test.iloc[1,2]))                    
                    PRO_BUY.append(confusion_train.iloc[1,2]/(confusion_train.iloc[1,1]+confusion_train.iloc[1,2]))                    
                    VAR.append(i)           
                else:
                    GN.append(-1)
                    TRAN.append(0)
                    MEDIANA.append(-1)
                    PRO.append((confusion_train.iloc[0,1]+confusion_train.iloc[1,2])/(confusion_train.iloc[0,2]+confusion_train.iloc[1,1]+confusion_train.iloc[0,1]+confusion_train.iloc[1,2]))
                    T_PRO.append((confusion_test.iloc[0,1]+confusion_test.iloc[1,2])/(confusion_test.iloc[0,2]+confusion_test.iloc[1,1]+confusion_test.iloc[0,1]+confusion_test.iloc[1,2]))
                    T_PRO_BUY.append(confusion_test.iloc[1,2]/(confusion_test.iloc[1,1]+confusion_test.iloc[1,2])) 
                    PRO_BUY.append(confusion_train.iloc[1,2]/(confusion_train.iloc[1,1]+confusion_train.iloc[1,2]))     
                    VAR.append(i) 
                                                  
                #h2o.remove(mod)
            except:
                ERROR.append(i)
                print("Error en: "+i)
         
        var_total=list(set(var_total) - set(ERROR))
        
        A=pd.DataFrame({"VAR":VAR,"PRO":PRO,"T_PRO":T_PRO,"GN":GN,"TRAN":TRAN,"MEDIANA":MEDIANA,"PRO_BUY":PRO_BUY,"T_PRO_BUY":T_PRO_BUY})        
        B=A[A.TRAN>=tran_min].sort_values(by_select,ascending=False)        
        pasos.append(B)
        #var_select.append(B.VAR.iloc[0])
                       
        var_total=list(set(var_total)-set([B.VAR.iloc[0]]))    
        if (B[by_select].iloc[0]-P2)>0.00005:
            P2=B[by_select].iloc[0]
            var_select.append(B.VAR.iloc[0])
        else:
            #h2o.shutdown()
            break        
        print(var_select)
          
    return var_select, pasos
