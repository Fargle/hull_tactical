import pandas as pd
import csv

results = pd.read_csv("results.csv")
account = pd.read_csv("account.csv")
prediction = results[len(results)-1]["prediction"]


buy = account['buy']
sell = account['sell']
available = account['available']

buys = {'max': 2500, 'med': 1000, 'min':500}


delta = results["prediction"].std()
if prediction > 0 and prediction < delta:
    if available > buys.get('med'):
        buy = buys.get('med')
    elif available > buys.get('min'):
        buy = buys.get('min')
    else:
        buy = available
    
elif prediction > 0 and prediction > delta:
    if available > buys.get('max'):
        buy = buys.get('max')
    elif available > buys.get('med'):
        buy = buy.get('med')
    elif available > buys.get('min'):
        buy = buy.get('min')

elif prediction < 0 and prediction < delta:
    if available > buys.get('med'):
        sell = buys.get('med')
    elif available > buys.get('min'):
        sell = buys.get('min')
    else:
        sell = available
    
elif prediction < 0 and prediction > delta:
    if available > buys.get('max'):
        sell = buys.get('max')
    elif available > buys.get('med'):
        sell = buy.get('med')
    elif available > buys.get('min'):
        sell = buy.get('min')

with open("account.csv", mode="w") as account:
    account_writer = csv.writer(account, delimiter = ",", quotechar='"', quoting=csv.QUOTE_NONE)
    account_writer.writerow(['buy', 'sell', 'available'])
    account_writer.writerow([buy, sell, available])
