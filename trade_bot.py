import pandas as pd
import csv
import matplotlib.pyplot as plt
import argparse
import os
import pickle
import numpy as np  
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", help="show the trading and selling through all data", action="store_true")
    parser.add_argument("--name", help="Name of model", type=str, default='model')
    parser.add_argument("--test", help ="test bot on results.csv", action="store_true")
    parser.add_argument("--data", type=str, help="Path to data", default=os.path.abspath("ucsbdata.csv"))
    return parser.parse_args()

class Test():
    def __init__(self, mean, std, cash=10000, market=0):
        self.cash = cash
        self.market = market
        self.mean = mean
        self.std = std


    def get_cash(self):
        return self.cash


    def get_market(self):
        return self.market


    def get_total(self):
        total = self.cash + self.market
        return max(0, total)


    def buy(self, amount):
        amount = max(0, amount)
        self.cash -= amount
        self.market += amount 
    

    def sell(self, amount):
        amount = max(0, amount)
        self.market -= amount
        self.cash += amount


    def calculate_interest(self, actual):
        actual = (actual*self.std) + self.mean
        self.market = max(0, self.market * (1 + (actual*0.1)))


def buy_or_sell(prediction, cash, market):
    buy = 0.0
    sell = 0.0
    if prediction >= 0:
        buy = cash
    else:
        sell = market
    return buy, sell


def get_available():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SAMPLE_SPREADSHEET_ID = '1x1-5f4ERy87WWWdaA3b4D8F7EqLlEpJuluJ9yOHqMS0'
    SAMPLE_RANGE_NAME = 'A2:L'

    creds = None

    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID,
                                range=SAMPLE_RANGE_NAME).execute()
    values = result.get('values', [])

    if not values:
        print('No data found.')
    else:
        df = pd.DataFrame(values)
        df.columns = ['no', 'Team Name', 'Cash', 'Market', 'Total', 'Prediction', 'Buy', 'Sell', 'Cash2', 'Market2', 'Total2', 'R']
        df = df.drop([0, 1])
        market_indx = df['Market2'].where(df['Team Name'] == 'bells in forts').last_valid_index()
        cash_indx = df['Cash2'].where(df['Team Name'] == 'bells in forts').last_valid_index()
        market = df['Market2'][market_indx]
        cash = df['Cash2'][cash_indx]
        if cash == '-':
            cash = '0'
        if market == '-':
            market = '0'
        return float(market.replace(',', '')), float(cash.replace(',', '').replace(' ', ''))


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


if __name__ == '__main__':
    args = setup()
    filename = args.name + "-results.csv"

    try:
        results = pd.read_csv(filename)
        original_data = pd.read_csv(args.data)
    except: 
        print("No results file found with name", filename)
        exit()

    mean = original_data['R'].mean()
    std = original_data['R'].std()
    results['prediction'] = (results['prediction']*std) + mean
    if args.test:
        results['actual'] = (results['actual']*std) + mean
        account = Test(mean, std)
        for i, j in results.iterrows():
            prediction = j['prediction']
            actual = j['actual']
            buy, sell = buy_or_sell(prediction, account.get_cash(), account.get_market())
            account.buy(amount=buy)
            account.sell(amount=sell)
            account.calculate_interest(actual)
            #print(account.get_total())
        print('TOTAL:' , account.get_total())
    else:
        prediction = float(results['prediction'][len(results)-1])
        market, cash = get_available()    
        buy, sell = buy_or_sell(prediction, cash, market)
        write_out('output.txt', prediction, int(buy), int(sell))
