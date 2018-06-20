"""
@author: Andres Fernando Guaca
"""

import h2o
import seleccion_variables as sv
import datos as dt
import funciones
import os

#h2o.init(max_mem_size=14) 
#BTC_STR""BTC_XEM","BTC_LTC""BTC_LTC",


def modelos_selecion_gbm(i="BTC_ETH",k=1400,s=9858,cb=1):
    datos=dt.data_coin(coin="BTC_ETH",periodo=14400,inicio_train="2017-04-01 00:00:00",pas=10,cb=cb).run()    
    var_total=list(set(datos[0].columns) - {"Tendencia",'date', 'high', 'low', 'close', 'volume', 'open', 'rt'})       
    for l in [0.01]:
        for d in [5,10]:
            for m in [10,20]:
                for t in [30,60]:
                    try:
                        from h2o.estimators.gbm import H2OGradientBoostingEstimator
                        mod=H2OGradientBoostingEstimator(nfolds=5,keep_cross_validation_predictions=True,learn_rate=l,max_depth=d,min_rows=m,ntrees=t)
                        seleccion=sv.seleccion_variables(mod,datos,"T_PRO",20,20,s,var_total)
                        funciones.save_object("resultados/{}/gbm_{}_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,l,s),seleccion)
                        h2o.remove_all()
                    except:
                        h2o.remove_all()
                        print("Error: "+"resultados/{}/gbm_{}_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,l,s))

                        
#h2o.init(max_mem_size=14)


modelos_selecion_gbm(i="BTC_ETH",k=14400,s=9389,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=2481,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=9938,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=4755,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=4621,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=7401,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=8947,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=6624,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=5976,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=7233,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=1777,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=8052,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=1174,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=3670,cb=6)
modelos_selecion_gbm(i="BTC_ETH",k=14400,s=9542,cb=6)


def modelos_selecion_random_forest(i="BTC_ETH",k=1400,s=9858,cb=1):
    datos=dt.data_coin(coin="BTC_ETH",periodo=14400,inicio_train="2017-04-01 00:00:00",pas=10,cb=cb).run()    
    var_total=list(set(datos[0].columns) - {"Tendencia",'date', 'high', 'low', 'close', 'volume', 'open', 'rt'})         
    for d in [5,10]:
        for m in [10,20]:
            for t in [30,60]:
                try:
                    if not os.path.isfile("resultados/{}/random_forest_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,s)):
                        
                        from h2o.estimators.random_forest import H2ORandomForestEstimator
                        mod=H2ORandomForestEstimator(nfolds=5,keep_cross_validation_predictions=True,max_depth=d,min_rows=m,ntrees=t)
                        seleccion=sv.seleccion_variables(mod,datos,"T_PRO",20,20,s,var_total)
                        funciones.save_object("resultados/{}/random_forest_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,s),seleccion)
                        
                    h2o.remove_all()
                    
                except:
                    h2o.remove_all()
                    print("Error: "+"resultados/{}/random_forest_{}_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,s))

                        
h2o.init(max_mem_size=14)

modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=9389,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=2481,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=9938,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=4755,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=4621,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=7401,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=8947,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=6624,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=5976,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=7233,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=1777,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=8052,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=1174,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=3670,cb=6)
modelos_selecion_random_forest(i="BTC_ETH",k=14400,s=9542,cb=6)


def modelos_selecion_xgboost(i="BTC_ETH",k=1400,s=9858,cb=1):
    datos=dt.data_coin(coin="BTC_ETH",periodo=14400,inicio_train="2017-04-01 00:00:00",pas=10,cb=cb).run()    
    var_total=list(set(datos[0].columns) - {"Tendencia",'date', 'high', 'low', 'close', 'volume', 'open', 'rt'})
    for l in [0.1,0.3]:         
        for d in [5,10]:
            for m in [10,20]:
                for t in [30,60]:
                    try:
                        if not os.path.isfile("resultados/{}/xgboost_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,l,s)):
                        
                            from h2o.estimators.xgboost import H2OXGBoostEstimator
                            mod=H2OXGBoostEstimator(nfolds=5,keep_cross_validation_predictions=True,max_depth=d,min_rows=m,ntrees=t)
                            seleccion=sv.seleccion_variables(mod,datos,"T_PRO",10,20,s,var_total)
                            funciones.save_object("resultados/{}/xgboost_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,s),seleccion)
                            
                        h2o.remove_all()
                        
                    except:
                        h2o.remove_all()
                        print("Error: "+"resultados/{}/xgboost_{}_{}_{}_{}_{}_{}_{}_{}".format(i,i,k,cb,d,m,t,s))

                   
h2o.init(max_mem_size=14)

#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=9389,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=2481,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=9938,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=4755,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=4621,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=7401,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=8947,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=6624,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=5976,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=7233,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=1777,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=8052,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=1174,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=3670,cb=6)
#modelos_selecion_xgboost(i="BTC_ETH",k=14400,s=9542,cb=6)



