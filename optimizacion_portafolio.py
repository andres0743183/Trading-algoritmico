"""
@author: Andres Fernando Guaca
"""

import datos as dt
import seleccion_variables as sv
import mercado
import h2o
import matplotlib.pyplot as plt

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
datos=dt.data_coin(coin="BTC_ETH",periodo=14400,inicio_train="2017-06-01 00:00:00",fin_train="2018-04-01 00:00:00", inicio_test="2018-04-01 00:00:00", fin_test="2018-06-01 00:00:00",pas=10,cb=6).run()
train = h2o.H2OFrame(datos[0])
test = h2o.H2OFrame(datos[1])
sv.tipificar_h2o(train)
splits = train.split_frame(ratios=[0.7], seed=9389)
precios=datos[1][["date","low","close"]].reset_index()
             
from h2o.estimators.random_forest import H2ORandomForestEstimator
mod2=H2ORandomForestEstimator(nfolds=5,keep_cross_validation_predictions=True,max_depth=5,min_rows=10,ntrees=60)
mod2.train(x=seleccion_rf[0:30], y="Tendencia", training_frame=train)
prediccion=mod2.predict(test).as_data_frame()
c=0

plt.xlabel('Número de transacciones ')
plt.ylabel('Utilidad')

for corte in [0.4,0.5,0.6]:

    prediccion["c_"+str(corte)]=[1 if i>corte else -1 for i in prediccion["p1"] ]
    c=c+1
    analisis=mercado.mercado_simulacion(precios,prediccion["c_"+str(corte)],stop=0.04,inversion=1,ct=0.002)
    analisis["GN"].cumsum().plot(figsize =(15,8))






































