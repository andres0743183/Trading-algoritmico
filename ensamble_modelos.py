"""
@author: Andres Fernando Guaca
"""

import datos as dt
import seleccion_variables as sv
import pandas as pd
import h2o


def prueba_modelos_ensamble(mod,train):
    modelos=[]
    semilla=[]
    T_PRO=[]
    for s in [431, 181,754,902,954,806, 400, 562, 552, 431]:
        splits = train.split_frame(ratios=[0.7], seed=s)
        
        pron=pd.DataFrame()
        nummod=1
        for m, v in mod:
            m.train(x=v, y="Tendencia", training_frame=splits[0])
          
            pron[str(nummod)]=m.predict(splits[1]).as_data_frame()["p1"]
            nummod=nummod+1
            
        pron["123"]=[1 if i>0.5 else -1 for i in (pron["1"]+pron["2"]+pron["3"])/3 ] 
        pron["12"]=[1 if i>0.5 else -1 for i in (pron["1"]+pron["2"])/2 ] 
        pron["13"]=[1 if i>0.5 else -1 for i in (pron["1"]+pron["3"])/2 ] 
        pron["23"]=[1 if i>0.5 else -1 for i in (pron["2"]+pron["3"])/2 ] 
        pron["1"]=[1 if i>0.5 else -1 for i in pron["1"] ]   
        pron["2"]=[1 if i>0.5 else -1 for i in pron["2"]] 
        pron["3"]=[1 if i>0.5 else -1 for i in pron["3"]] 
        pron["Tendencia"] =splits[1].as_data_frame()["Tendencia"]     
              
        semilla.append(s)
        modelos.append("(1)")
        con1=pd.crosstab(pron["1"],pron["Tendencia"])
        T_PRO.append((con1.iloc[0,0]+con1.iloc[1,1])/(con1.iloc[0,1]+con1.iloc[0,0]+con1.iloc[1,0]+con1.iloc[1,1]))        
        semilla.append(s)
        modelos.append("(2)")
        con1=pd.crosstab(pron["2"],pron["Tendencia"])
        T_PRO.append((con1.iloc[0,0]+con1.iloc[1,1])/(con1.iloc[0,1]+con1.iloc[0,0]+con1.iloc[1,0]+con1.iloc[1,1]))                
        semilla.append(s)
        modelos.append("(3)")
        con1=pd.crosstab(pron["3"],pron["Tendencia"])
        T_PRO.append((con1.iloc[0,0]+con1.iloc[1,1])/(con1.iloc[0,1]+con1.iloc[0,0]+con1.iloc[1,0]+con1.iloc[1,1]))        
        semilla.append(s)
        modelos.append("(1, 2)")
        con1=pd.crosstab(pron["12"],pron["Tendencia"])
        T_PRO.append((con1.iloc[0,0]+con1.iloc[1,1])/(con1.iloc[0,1]+con1.iloc[0,0]+con1.iloc[1,0]+con1.iloc[1,1]))
        semilla.append(s)
        modelos.append("(1, 3)")
        con1=pd.crosstab(pron["13"],pron["Tendencia"])
        T_PRO.append((con1.iloc[0,0]+con1.iloc[1,1])/(con1.iloc[0,1]+con1.iloc[0,0]+con1.iloc[1,0]+con1.iloc[1,1]))
        semilla.append(s)
        modelos.append("(2, 3)")
        con1=pd.crosstab(pron["23"],pron["Tendencia"])
        T_PRO.append((con1.iloc[0,0]+con1.iloc[1,1])/(con1.iloc[0,1]+con1.iloc[0,0]+con1.iloc[1,0]+con1.iloc[1,1]))      
        semilla.append(s)
        modelos.append("(1, 2, 3)")
        con1=pd.crosstab(pron["123"],pron["Tendencia"])
        T_PRO.append((con1.iloc[0,0]+con1.iloc[1,1])/(con1.iloc[0,1]+con1.iloc[0,0]+con1.iloc[1,0]+con1.iloc[1,1]))               
       
    return pd.DataFrame({"modelos":modelos,"semilla":semilla,"T_PRO":T_PRO})


seleccion_gbm=['rt_9', 'volume_3', 'rsi_0', 'macd2_1', 'CDLRICKSHAWMAN', 'macd1_1', 'rt_6', 'macd1_2', 'rt_1', 'volume_8', 'macd3_2', 'volume_7',
 'rt_3', 'macd3_1', 'volume_4', 'CDLBELTHOLD', 'CDLHIGHWAVE', 'rsi_3', 'rt_4', 'CDLDOJI', 'rt_2', 'rt_5', 'macd3_26', 'volume_5',
 'rt_7', 'volume_6', 'CDLSHORTLINE', 'volume_2', 'CDLENGULFING', 'rsi_2', 'rt_8', 'CDLHIKKAKE', 'rsi_1', 'volume_9', 'volume_1',
 'CDL3WHITESOLDIERS', 'macd2_2', 'CDLMARUBOZU', 'CDLHARAMI', 'CDLSPINNINGTOP', 'CDLCLOSINGMARUBOZU', 'CDLLONGLINE', 'CDLHAMMER',
 'CDLDOJISTAR', 'CDL3LINESTRIKE', 'CDLMORNINGDOJISTAR', 'CDLDARKCLOUDCOVER', 'CDL3BLACKCROWS', 'CDLIDENTICAL3CROWS', 'CDLEVENINGDOJISTAR',
 'CDLHANGINGMAN', 'CDLINVERTEDHAMMER', 'CDLMORNINGSTAR', 'CDL3OUTSIDE', 'CDLMATCHINGLOW', 'CDLRISEFALL3METHODS', 'CDLSHOOTINGSTAR', 
 'CDLADVANCEBLOCK', 'CDLSEPARATINGLINES', 'CDLUNIQUE3RIVER', 'CDLSTALLEDPATTERN']

seleccion_rf=['macd1_2', 'volume_1', 'volume_5', 'CDLMORNINGSTAR', 'macd3_26', 'rsi_2', 'CDLLONGLINE', 'macd2_1', 'rt_8', 'rt_5',
 'CDLSHOOTINGSTAR', 'CDLLADDERBOTTOM', 'macd3_1', 'CDLSPINNINGTOP', 'CDLGAPSIDESIDEWHITE', 'CDLTRISTAR', 'CDLXSIDEGAP3METHODS',
 'rt_7', 'CDL3LINESTRIKE', 'rt_4', 'CDLRISEFALL3METHODS', 'CDL3INSIDE', 'CDLHARAMICROSS', 'CDLDOJISTAR', 'CDLDRAGONFLYDOJI',
 'CDLEVENINGDOJISTAR', 'macd2_2', 'volume_2', 'CDLSEPARATINGLINES', 'CDLMORNINGDOJISTAR', 'CDLEVENINGSTAR', 'rt_6', 'CDLTAKURI',
 'rt_2', 'CDLENGULFING', 'volume_4', 'CDLMATCHINGLOW', 'volume_7', 'macd1_1', 'volume_8', 'rsi_3', 'rsi_1', 'volume_9', 'CDLIDENTICAL3CROWS',
 'CDLCLOSINGMARUBOZU', 'CDLDARKCLOUDCOVER', 'CDLRICKSHAWMAN', 'CDLMARUBOZU', 'rt_1', 'CDL3BLACKCROWS', 'CDL3OUTSIDE', 'CDLADVANCEBLOCK',
 'rsi_0', 'CDLHAMMER', 'CDLDOJI', 'macd3_2', 'volume_6', 'CDLBELTHOLD', 'CDLSHORTLINE', 'CDLHIGHWAVE', 'CDLGRAVESTONEDOJI', 'CDLSTALLEDPATTERN',
 'CDLHIKKAKE', 'CDLINVERTEDHAMMER', 'volume_3', 'rt_3', 'CDLTASUKIGAP', 'rt_9', 'CDLHIKKAKEMOD', 'CDLUNIQUE3RIVER', 'CDLHARAMI',
 'CDL3WHITESOLDIERS', 'CDLHOMINGPIGEON', 'CDLHANGINGMAN']

h2o.init(max_mem_size=14) 
datos=dt.data_coin(coin="BTC_ETH",periodo=14400,inicio_train="2017-06-01 00:00:00",fin_train="2018-04-01 00:00:00", inicio_test="2018-04-01 00:00:00", fin_test="2018-05-28 00:00:00",pas=10,cb=6).run()
train = h2o.H2OFrame(datos[0])
test = h2o.H2OFrame(datos[1])
sv.tipificar_h2o(train)
    
from h2o.estimators.gbm import H2OGradientBoostingEstimator
mod1=H2OGradientBoostingEstimator(nfolds=5,keep_cross_validation_predictions=True,learn_rate=0.01,max_depth=10,min_rows=10,ntrees=30)
                       
from h2o.estimators.random_forest import H2ORandomForestEstimator
mod2=H2ORandomForestEstimator(nfolds=5,keep_cross_validation_predictions=True,max_depth=5,min_rows=10,ntrees=60)
                  
from h2o.estimators.xgboost import H2OXGBoostEstimator
mod3=H2OXGBoostEstimator(nfolds=5,keep_cross_validation_predictions=True,max_depth=5,min_rows=10,ntrees=60)

modelos=[(mod1, seleccion_gbm[0:30]),(mod2,seleccion_rf[0:30]),(mod3,seleccion_rf[0:30])]
res=prueba_modelos_ensamble(modelos,train)
res.boxplot("T_PRO",by=[ "modelos"],figsize =(15,8),showmeans=True,patch_artist=False,vert=True)


















