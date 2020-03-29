import pandas as pd
import csv
import matplotlib.pyplot as plt

buys = {'max': 2500, 'med': 1000, 'min':500}

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", help="show the trading and selling through all data", action="store_true")
    parser.add_argument("--data", type=str, help="Path to data", default=os.path.abspath("ucsbdata.csv"))
    return parser.parse_args()

def buy_or_sell(available, prediction, delta):
    buy = 0
    sell = 0
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

    return (buy, sell)

def determine_available(buy, sell, available, r):
    available = available - buy
    available = available + (sell * (1 + r))

    return available

def write_account(buy, sell, available):
    with open("account.csv", mode="w") as account:
        account_writer = csv.writer(account, delimiter = ",", quotechar='"', quoting=csv.QUOTE_NONE)
        account_writer.writerow(['buy', 'sell', 'available'])
        account_writer.writerow([buy, sell, available])

if __name__ == '__main__':
    args = setup()
    results = pd.read_csv("results.csv")
    account = pd.read_csv("account.csv")


    delta = results["prediction"].std()
    buy = account['buy']
    sell = account['sell']


    if args.show:
        plt.figure()
        for row in results:
            available = determine_available()
        plt.plot()
    prediction = results[len(results)-1]["prediction"]
    r = results[len(results)-1]["actual"]

    
    available = determine_available(buy, sell)
    buy_or_sell(available, prediction, delta)
