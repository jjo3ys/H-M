import pandas as pd
import numpy as np
import csv

from tqdm import tqdm
tqdm.pandas()

def get_ldbw():
    df = pd.read_csv("transactions_train.csv", usecols=['t_dat', 'customer_id', 'article_id'], dtype={'article_id':str})
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    last_ts = df['t_dat'].max()
    #print(last_ts)#2020-09-22 00:00:00

    df['ldbw'] = df['t_dat'].progress_apply(lambda d: last_ts-(last_ts-d).floor('7D'))#transactions.csv에서 가장 마지막 날짜를 기준으로 7일 씩 그룹화
    df.to_csv("last_day_of_billing_week.csv", index=None)

def get_weekly():
    df = pd.read_csv("last_day_of_billing_week.csv", index_col='Unnamed: 0')

    # last_ts = pd.to_datetime(df['t_dat']).max()
    weekly_sales = df.drop("customer_id", axis=1).groupby(['ldbw', 'article_id']).count()
    # print(weekly_sales)
    weekly_sales = weekly_sales.rename(columns={'t_dat': 'count'})
    weekly_sales.to_csv("weekly_sales.csv")
    # print(weekly_sales)
    # df = df.join(weekly_sales, on=['ldbw', 'article_id'])

    # weekly_sales = weekly_sales.reset_index().set_index('article_id')
    # last_day = last_ts.strftime('%Y-%m-%d')

    # df = df.join(weekly_sales.loc[weekly_sales['ldbw']==last_day, ['count']], on='article_id', rsuffix='_targ')
    # df['count_targ'].fillna(0, inplace=True)
    # del weekly_sales

    # df['quotient'] = df['count_targ'] / df['count']
    # df.to_csv("weekly_sales.csv")

def get_weekly_dict():
    df = pd.read_csv("weekly_sales.csv").to_numpy()
    ws = {}

    for d in df:
        try:
            ws[d[0]][d[1]] = d[2]
        except:
            ws[d[0]] = {}
            ws[d[0]][d[1]] = d[2] 

    with open("weekly_sales_dict.csv", 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow['t_dat', 'weekly_sales_count', 'weekly_top_sales']
        for w in tqdm(ws.items()):
            sales = sorted(w[1].items(), key=lambda x: -x[1])
            wr.writerow([w[0], len(sales), sales])

def get_submission():
    df = pd.read_csv("last_day_of_billing_week.csv", index_col='Unnamed: 0')
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    last_ts = df['t_dat'].max()
    weekly_sales = pd.read_csv("weekly_sales.csv")
    df = pd.merge(df, weekly_sales, on=['ldbw', 'article_id'])
    print(df)

    weekly_sales = weekly_sales.reset_index().set_index('article_id')
    print(weekly_sales)
    last_day = last_ts.strftime('%Y-%m-%d')

    df = df.join(weekly_sales.loc[weekly_sales['ldbw']==last_day, ['count']], on='article_id', rsuffix='_targ')
    df['count_targ'].fillna(0, inplace=True)
    del weekly_sales

    df['quotient'] = df['count_targ'] / df['count']

    target_sales = df.drop('customer_id', axis=1).groupby('article_id')['quotient'].sum()
    general_pred = target_sales.nlargest(12).index.tolist()
    del target_sales

    purchase_dict = {}

    for i in tqdm(df.index):
        cust_id = df.at[i, 'customer_id']
        art_id = df.at[i, 'article_id']
        t_dat = df.at[i, 't_dat']

        if cust_id not in purchase_dict:
            purchase_dict[cust_id] = {}

        if art_id not in purchase_dict[cust_id]:
            purchase_dict[cust_id][art_id] = 0
        
        x = max(1, (last_ts - t_dat).days)

        a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
        y = a / np.sqrt(x) + b * np.exp(-c*x) - d

        value = df.at[i, 'quotient'] * max(0, y)
        purchase_dict[cust_id][art_id] += value

    with open('purchase_dict.csv', 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        for item in purchase_dict.items():
            wr.writerow([wr[0], wr[1]])

# sub = pd.read_csv('sample_submission.csv')

# pred_list = []
# for cust_id in tqdm(sub['customer_id']):
#     if cust_id in purchase_dict:
#         series = pd.Series(purchase_dict[cust_id])
#         series = series[series > 0]
#         l = series.nlargest(12).index.tolist()
#         if len(l) < 12:
#             l = l + general_pred[:(12-len(l))]
#     else:
#         l = general_pred
#     pred_list.append(' '.join(l))

# sub['prediction'] = pred_list
# sub.to_csv('submission.csv', index=None)
def get_weekly_list(day_of_the_end):
    df = pd.read_csv("last_day_of_billing_week.csv", index_col='Unnamed: 0')
    df = df.loc[df['ldbw']==day_of_the_end, ['customer_id', 'article_id']].to_numpy()
    result_dict = {}
    for d in df:
        if d[0] not in result_dict:
            result_dict[d[0]] = [d[1]]
        else:
            result_dict[d[0]].append(d[1])

    return result_dict

a = get_weekly_list('2020-09-22')
print(a)

def get_score(val_list, predict_list):
    answer = 0
    for val in val_list:
        if val in predict_list:
            answer += 1
    val_len = len(val_list):
    