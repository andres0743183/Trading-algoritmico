"""
@author: Andres Fernando Guaca
"""

import pandas as pd
import talib
import inspect
import re 
import zigzag
import numpy  as np

class data_coin(object):   
    def __init__(self, coin="BTC_LTC",periodo=7200,
                inicio_train="2016-06-01 00:00:00", fin_train="2018-04-01 00:00:00",
                inicio_test="2018-04-01 00:00:00", fin_test="2018-05-28 00:00:00",pas=15,cb=5):
        """
        Esta clase realiza la lectura de datos desde https://poloniex.com/ y crea  los datos  necesarios  para este proyecto.
        coin: valor de la criptomoneda a analizar respecto de otra (por ejemplo:  BTC_LTC, BTC_ETC, etc.)
        periodo: periodo de los datos a analizar (valores válidos: 300, 900, 1800, 7200, 14400 y 86400).
        inicio_train: fecha de inicio de los datos de entrenamiento.
        fin_train: fecha de fin de los datos de entrenamiento.
        inicio_test: fecha de inicio de los datos de prueba.
        fin_test: fecha de fin de los datos de prueba.
        pas: número de retardos para los datos de retorno y volumen.
        Cb: cantidad de valores absolutos de la media que se toman como objetivo de la tendencia.
        """
                 
        self.coin=coin
        self.pas=pas
        self.inicio_train=inicio_train
        self.fin_train=fin_train
        self.inicio_test=inicio_test
        self.fin_test=fin_test 
        self.cb=cb                      
        
        self.file="https://poloniex.com/public?command=returnChartData&currencyPair="+coin+\
        "&start="+str(1451610000)+"&end=9999999999&period="+str(periodo)    

    def argumentos(self,funcion):        
        reg =re.compile( "^ *" + funcion.__name__+ "\(([^)]+)\).*",\
                        flags = re.DOTALL | re.MULTILINE).match(funcion.__doc__).group(1)    
        arg=re.sub('[\[ ?=\]]', '', reg)
        arg=re.split('[,]+',arg, flags=re.IGNORECASE) 
        return arg   
    
    def contructor(self,funcion,maximo=50,salto=2):
        arg=self.argumentos(eval('talib.'+funcion))
 
        constante=[];variable=[];fun=[]

        for i in arg:
            if i=='real' or i=='open' or i=='high' or i=='low'  or i=='close' or i=='volume':
                constante.append(i)
            elif i== 'fastperiod' or i=='slowperiod' or i=='timeperiod':
                variable.append(i)
        texcos=[str(x) for x in constante]
        if len(variable)==1:
            texvar=[str(variable[0])+'='+str(x) for x in range(2,maximo,salto)]
        elif len(variable)==2:
            texvar=[]
            for i in range(2,maximo,salto):         
                for k in range(i,maximo,salto):
                    texvar.append(str(variable[0])+'='+str(i)+','+str(variable[1])+'='+str(k))
        else:
            texvar=[]                   
        if len(texvar)!=0:
            fun=['talib.'+funcion+'('+', '.join(texcos+[i])+')' for i in texvar]            
        else:
            fun=['talib.'+funcion+'('+', '.join(texcos)+')']
        
        return fun
    
    def objetivo(self,datos):
        close=datos["close"].as_matrix()
        cambio=datos["close"].pct_change().abs().mean()*self.cb
        pivots=zigzag.peak_valley_pivots(close, cambio, cambio*-1)
        movimiento=zigzag.pivots_to_modes(pivots)
        return movimiento
    
    
    def indicadores(self,datos):
        
        rt=pd.DataFrame(datos["close"].pct_change()).rename(columns={"close":"rt"})
        obj=pd.DataFrame(self.objetivo(datos)).rename(columns={0:"Tendencia"})
        salida = pd.concat([datos[["date","high","low","close","volume","open"]], rt,obj], axis=1)
                            
        for  i in range(1,self.pas):            
            resultado=salida[["rt","volume"]].shift(i) ##[["rt","high","low","close","volume"]]
            columnas=list(resultado.columns)
            nombres=[columnas[k]+"_"+str(i) for k in range(len(columnas))]
            resultado.columns=nombres
            salida = pd.concat([salida, resultado], axis=1)
            
        high=datos["high"].as_matrix()
        low=datos["low"].as_matrix()
        close=datos["close"].as_matrix()
        real=datos["close"].as_matrix()
        volume=datos["volume"].as_matrix()
        open=datos["open"].as_matrix()
        open, high, low,close, real, volume
        indicadores=inspect.getmembers(talib)[165][1]['Pattern Recognition']#[0:0]
        
        for f in indicadores:
            cal_ind=pd.DataFrame(eval("talib."+f+"(open, high, low,close)")).shift(1)
            cal_ind.columns=[f]
            salida=pd.concat([salida, cal_ind], axis=1)
            
           
        rsi=self.contructor("RSI")           
        for f in range(0,len(rsi)):
            cal_ind=pd.DataFrame(eval(rsi[f])).shift(1)
            cal_ind.columns=["rsi_"+str(f)]
            salida=pd.concat([salida, cal_ind], axis=1)         
        
        emac=self.contructor("MACD") 
        for f in range(0,len(emac)):
            cal_ind=pd.DataFrame(eval(emac[f])[0]).shift(1)
            cal_ind.columns=["macd1_"+str(f)]
            cal_ind_1=pd.DataFrame(eval(emac[f])[1]).shift(1)
            cal_ind_1.columns=["macd2_"+str(f)]
            cal_ind_2=pd.DataFrame(eval(emac[f])[2]).shift(1)
            cal_ind_2.columns=["macd3_"+str(f)]
            salida=pd.concat([salida, cal_ind,cal_ind_1,cal_ind_2], axis=1)  
       
        correlacion=salida.corr()
        borrar=[]
        
        for i in range(len(correlacion.columns)):
            for k in range(i,len(correlacion.columns)):
                       
                if np.isnan(correlacion.iloc[i,k]) and i==k:
                    borrar.append(correlacion.columns[k])
                    
                else:
                    if i!=k and correlacion.iloc[i,k]>0.99:
                        borrar.append(correlacion.columns[k])
                            
        salida=salida[list((set(salida.columns)-set(borrar)) | {"Tendencia",'date', 'high', 'low', 'close', 'volume', 'open', 'rt'})]           
        return salida[(salida.date>=self.inicio_train) & (salida.date<self.fin_train)], salida[(salida.date>=self.inicio_test) & (salida.date<=self.fin_test)]

               
    def run(self):
        datos = pd.read_json(self.file)
        return self.indicadores(datos)
      
        
#A=data_coin().run()
#real=A[0]["open"].as_matrix()    
