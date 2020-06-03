# from statsmodels.tsa.statespace.sarimax import SARIMAX
# def sarimaxmodel(row):
#     model=SARIMAX(row,order=(1,1,1),seasonal_order=(1,1,1,1))
#     model_fit=model.fit(disp=False)
#     yhat =model_fit.predict(len(row),len(row))
#     return yhat

#
#
from statsmodels.tsa.ar_model import AR

def analysismodel(row):
    model=AR(row)
    model_fit=model.fit()
    yhat=model_fit.predict(len(row),len(row))
    return yhat

#
# from statsmodels.tsa.arima_model import ARMA
#
# def armamodel(row):
#     model=ARMA(row,order=(2,1))
#     model_fit=model.fit(disp=False)
#     yhat=model_fit.predict(len(row),len(row))
#     return yhat



# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# def exponmodel(row):
#     model=ExponentialSmoothing(row)
#     model_fit=model.fit()
#     yhat=model_fit.predict(len(row),len(row))
#     return yhat