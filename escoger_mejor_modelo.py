"""
@author: Andres Fernando Guaca
"""

import funciones
import pandas as pd
import os 


def resultados_gbm(i="BTC_ETH",k=14400,cb=6):   
    salida=[]
    learn_rate=[]
    max_depth=[]
    min_rows=[]
    ntrees=[]
    semilla=[]
    for s in [9389,2481,9938,4755,4621,7401,8947,6624,5976,7233,1777,8052,1174,3670,9542]:
        for l in [0.01]:
            for d in [5,10]:
                for m in [10,20]:
                    for t in [30,60]:
                        if os.path.isfile("resultados/{}/gbm_{}_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,l,s)):
                            learn_rate.append(l)
                            max_depth.append(d)
                            min_rows.append(m)
                            ntrees.append(t)
                            semilla.append(s)
                            A=funciones.load_objeto("resultados/{}/gbm_{}_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,l,s))
                            salida.append(A[1][len(A[1])-1].head(1)[["PRO","T_PRO","T_PRO_BUY"]])

    resultados=pd.concat(salida)
    resultados["learn_rate"]=learn_rate
    resultados["max_depth"]=max_depth
    resultados["min_rows"]=min_rows
    resultados["ntrees"]=ntrees
    resultados["semilla"]=semilla
    return resultados


resultados=resultados_gbm()

print(resultados[["learn_rate",  "max_depth",  "min_rows",  "ntrees","PRO","T_PRO"]].to_latex())
resultados.boxplot("T_PRO",by=[ "max_depth",  "min_rows",  "ntrees"],figsize =(15,8),showmeans=True,patch_artist=False,vert=True)
resultados.boxplot("T_PRO_BUY",by=[ "max_depth",  "min_rows",  "ntrees"],figsize =(15,8))



def variables_gbm(i="BTC_ETH",k=14400,cb=6,l=0.01,d=5,m=10,t=30):   
    salida=[]
    salida_2=[]
    var_total=[]
    for s in [9389,2481,9938,4755,4621,7401,8947,6624,5976,7233,1777,8052,1174,3670,9542]:                    #print("{}_{}_{}_{}_{}_{}".format(i,k,d,m,t,l)) 
        if os.path.isfile("resultados/{}/gbm_{}_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,l,s)):
            A=funciones.load_objeto("resultados/{}/gbm_{}_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,l,s))
            salida.append(A[0])
            var_total=var_total+A[0]
    frecuencia= [var_total.count(w) for w in set(var_total)] # a list comprehension
    salida_2=pd.DataFrame()
    salida_2["Variable"]=list(set(var_total))
    salida_2["Frecuencia"]=frecuencia        
    return salida,salida_2.sort_values("Frecuencia",ascending=False)

seleccion_gbm=variables_gbm()
print(seleccion_gbm[1].reset_index()[["Variable",  "Frecuencia"]].to_latex())


def resultados_rf(i="BTC_ETH",k=14400,cb=6):   
    salida=[]
    max_depth=[]
    min_rows=[]
    ntrees=[]
    semilla=[]
    for s in [9389,2481,9938,4755,4621,7401,8947,6624,5976,7233,1777,8052,1174,3670,9542]:
        for d in [5,10]:
            for m in [10,20]:
                for t in [30,60]:
                    if os.path.isfile("resultados/{}/random_forest_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,s)):

                        max_depth.append(d)
                        min_rows.append(m)
                        ntrees.append(t)
                        semilla.append(s)
                        A=funciones.load_objeto("resultados/{}/random_forest_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,s))
                        salida.append(A[1][len(A[1])-1].head(1)[["PRO","T_PRO","T_PRO_BUY"]])

    resultados=pd.concat(salida)
    resultados["max_depth"]=max_depth
    resultados["min_rows"]=min_rows
    resultados["ntrees"]=ntrees
    resultados["semilla"]=semilla
    return resultados


resultados=resultados_rf()

print(resultados[[ "max_depth",  "min_rows",  "ntrees","PRO","T_PRO"]].to_latex())
resultados.boxplot("T_PRO",by=[ "max_depth",  "min_rows",  "ntrees"],figsize =(15,8),showmeans=True,patch_artist=False,vert=True)
resultados.boxplot("T_PRO_BUY",by=[ "max_depth",  "min_rows",  "ntrees"],figsize =(15,8))


def variables_rf(i="BTC_ETH",k=14400,cb=6,l=0.01,d=5,m=10,t=30):   
    salida=[]
    salida_2=[]
    var_total=[]
    for s in [9389,2481,9938,4755,4621,7401,8947,6624,5976,7233,1777,8052,1174,3670,9542]:                    #print("{}_{}_{}_{}_{}_{}".format(i,k,d,m,t,l)) 
        if os.path.isfile("resultados/{}/random_forest_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,s)):
            A=funciones.load_objeto("resultados/{}/random_forest_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,s))
            salida.append(A[0])
            var_total=var_total+A[0]
    frecuencia= [var_total.count(w) for w in set(var_total)] # a list comprehension
    salida_2=pd.DataFrame()
    salida_2["Variable"]=list(set(var_total))
    salida_2["Frecuencia"]=frecuencia        
    return salida,salida_2.sort_values("Frecuencia",ascending=False)

seleccion_rf=variables_rf()
print(seleccion_rf[1].reset_index()[["Variable",  "Frecuencia"]].to_latex())

