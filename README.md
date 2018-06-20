# Trading algorítmico en criptomonedas

Este repositorio contiene la información de las  funciones creadas para el trabajo  fin de máster denominado “Trading algorítmico en criptomonedas”.

En resumen el trabajo consiste en el desarrollo de un algoritmo de trading para la construcción de una cartera con la criptomondea Bitcoin-Ethereum. Para ello, se trabaja a partir de datos obtenidos desde la plataforma Poloniex https://poloniex.com/. A partir de estos se sigue un procedimiento clásico en la construcción de un trading algorítmico típico de los mercados financieros, partiendo de un análisis preliminar para continuar con la modelización y evaluación del desempeño de modelos de alta capacidad predictiva: Random forest, Gradient Boosting y Extreme gradient boosting. La modelización mediante dichas técnicas consiste en la generación de señales de mercado ante cambios en la tendencia del precio y las mismas fueron probadas en un ambiente de simulación de mercado real obteniendo mejorando los rendimientos.

En el archivo “resultados_seleccion_variables.py” se muestra un ejemplo de como utilizar las funciones “seleccion_variables” y “data_coin”, los  resultados de correr este archivo Python, crea una serie de resultados que se guardan en “resultados/ETH/res”.

Para leer estos resultados del apartado  anterior  en el archivo  “escoger_mejor_modelo.py”, se muestra un ejemplo  de lectura y la realización de un resumen gráfico. 


