import pandas as pd
import csv
import matplotlib.pyplot as plt
import argparse
import os
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

buys = {'max': 2500, 'med': 1000, 'min': 500}

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", help="show the trading and selling through all data", action="store_true")
    parser.add_argument("--new", help="make a new account", action="store_true")
    parser.add_argument("--testing", help ="test bot on results.csv", action="store_true")
    parser.add_argument("--data", type=str, help="Path to data", default=os.path.abspath("ucsbdata.csv"))
    return parser.parse_args()

def buy_or_sell(prediction, cash, market):
    buy = 0
    sell = 0 
    if prediction >= 0:
        buy = cash
    else:
        sell = market
    return buy, sell

def get_available():
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SAMPLE_SPREADSHEET_ID = '1x1-5f4ERy87WWWdaA3b4D8F7EqLlEpJuluJ9yOHqMS0'
    SAMPLE_RANGE_NAME = 'A2:L'

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
        print('')
        df = pd.DataFrame(values)
        df.columns = ['no', 'Team Name', 'Cash', 'Market', 'Total', 'Prediction', 'Buy', 'Sell', 'Cash2', 'Market2', 'Total2', 'R']
        df = df.drop([0, 1])
        market_indx = df['Market2'].where(df['Team Name'] == 'bells in forts').last_valid_index()
        cash_indx = df['Cash2'].where(df['Team Name'] == 'bells in forts').last_valid_index()
        market = df['Market2'][market_indx]
        cash = df['Cash2'][cash_indx] 

        return market, cash


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


if __name__ == '__main__':
    args = setup()
    results = pd.read_csv("results.csv")
    prediction = results['prediction'][len(results)-1]
    cash, market = get_available()
    buy, sell = buy_or_sell(prediction, cash, market)
    write_out('output.txt', prediction, buy, sell)



