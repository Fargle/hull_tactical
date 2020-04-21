import pandas as pd
import csv
import matplotlib.pyplot as plt
import argparse
import os

buys = {'max': 2500, 'med': 1000, 'min': 500}

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", help="show the trading and selling through all data", action="store_true")
    parser.add_argument("--new", help="make a new account", action="store_true")
    parser.add_argument("--testing", help ="test bot on results.csv", action="store_true")
    parser.add_argument("--data", type=str, help="Path to data", default=os.path.abspath("ucsbdata.csv"))
    return parser.parse_args()

def buy_or_sell(available, invested, prediction, delta):
    buy = 0
    sell = 0
    #assert(invested >= 0)

    if prediction > 0 and prediction < delta:
        if available >= buys.get('med'):
            buy = buys.get('med')
        elif available >= buys.get('min'):
            buy = buys.get('min')
        else:
            buy = available
    
    elif prediction > 0 and prediction > delta:
        if available >= buys.get('max'):
            buy = buys.get('max')
        elif available >= buys.get('med'):
            buy = buys.get('med')
        elif available >= buys.get('min'):
            buy = buys.get('min')

    elif prediction < 0 and abs(prediction) < delta:
        if invested >= buys.get('med'):
            sell = buys.get('med')
        elif invested >= buys.get('min'):
            sell = buys.get('min')
    
    elif prediction < 0 and abs(prediction) > delta:
        if invested >= buys.get('max'):
            sell = buys.get('max')
        elif invested >= buys.get('med'):
            sell = buys.get('med')
        elif invested >= buys.get('min'):
            sell = buys.get('min')

    return buy, sell

def buy_stock(buy, available, invested):
    assert(buy > 0)
    assert(available - buy >= 0)
    available = available - buy
    invested = invested + buy
    return available, invested 


def sell_stock(sell, available, invested):
    assert(sell > 0)
    assert(invested - sell >= 0)
    available = available + sell
    invested = invested - sell
    return available, invested


def calculate_earnings(r, invested):
    invested = invested * (1 + r)
    return invested


def write_out(file, prediction, buy, sell):
    with open(file, mode='w') as output:
        output.write(str(prediction) + "\n")
        output.write(str(buy) + "\n")
        output.write(str(sell) + "\n")


def write_csv(file, available, invested, total):
    with open(file, mode="w", newline='', encoding='utf-8') as account:
        account_writer = csv.writer(account, delimiter = ",", quotechar='"', quoting=csv.QUOTE_NONE)
        account_writer.writerow(['available', 'invested', 'total'])
        account_writer.writerow([available, invested, total])


def read_csv(file):
    with open(file, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        for row in csv_reader:
            pass
    return row

available = 10000
invested = 10000
total = 0

if __name__ == '__main__':
    args = setup()
    results = pd.read_csv("results.csv")

    if args.testing:
        testing = args.testing

        if args.new:
            write_csv(file='account.csv', available=10000, invested=0, total=10000)

        account = read_csv("account.csv")
        available = float(account['available'])
        invested = float(account['invested'])
        total = float(account['total'])
        r = results.iloc[len(results)-1]["actual"]
        invested = calculate_earnings(r, invested)

    delta = results["actual"].std()
    prediction = results.iloc[len(results)-1]["prediction"].astype(float)
    buy, sell = buy_or_sell(available, invested, prediction, delta)
    write_out("output.txt", prediction, buy, sell)

    if args.show:
        try:
            df = pd.read_csv("ucsbdata.csv")   
            df.dropna()
            del df['Index']
            df = df.astype(float)
        except:
            print("ucsbdata not found")
        try:
            plt.figure()
            plt.plot(df["OPEN"])
        except:
            print('you got some problems with matplot')

        buy = 0
        sell = 0
        for index, row in results.iterrows():
            prediction = row["prediction"]
            r = row['actual']
            invested = calculate_earnings(r, invested)
            print('prediction:', str(round(prediction, 2)), 'r:', str(round(r, 2)), 'buy:', buy, 'sell:', sell, 'invested', invested)
            buy, sell = buy_or_sell(available, invested, prediction, delta)
            if buy != 0:
                available, invested = buy_stock(buy, available, invested)
            elif sell != 0: 
                available, invested = sell_stock(sell, available, invested)

        total = invested + available

