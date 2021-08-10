import yfinance as yf
import datetime
import logging
from fbprophet import Prophet
from statsmodels.tsa import holtwinters
  
logging.basicConfig(filename="/tmp/newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger=logging.getLogger()
logger.setLevel(logging.DEBUG)


def prophet_forecast(stock, days_to_get, value_to_get):
	
	now = datetime.datetime.today()
	date = now.strftime("%Y-%m-%d")
	delta = datetime.timedelta(days=120)
	start = now - delta
	start = start.strftime("%Y-%m-%d")
	data = yf.download(stock, start=start, end=date)

	data['ds'] = data.index
	data['y'] = data[value_to_get]
	df = data[['ds', 'y']]

	logger.info("Modeling the data using Prophet")
	model = Prophet(seasonality_mode="multiplicative", daily_seasonality=True)
	model.fit(df)
	future = model.make_future_dataframe(periods=days_to_get)
	logger.info("Making predictions")
	forecast = model.predict(future)

	return forecast[['ds', 'yhat']]

def exp_smoothing_forecast(stock, days_to_get, value_to_get):

  now = datetime.datetime.today()
  date = now.strftime("%Y-%m-%d")
  delta = datetime.timedelta(days=45)
  start = now - delta
  start = start.strftime("%Y-%m-%d")
  data = yf.download(stock, start=start, end=date)

  data['date'] = date.index
  data.index = list(range(0, data.shape[0]))
  model = holtwinters.ExponentialSmoothing(data[value_to_get], seasonal='multiplicative', 
                              trend='multiplicative', seasonal_periods=7)
  logger.info("Model fitting")
  model_fit = model.fit()
  logger.info("Forecasting values")
  values = model_fit.forecast(days_to_get)
  return values.values
