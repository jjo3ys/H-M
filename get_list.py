import pandas as pd
import numpy as np
import csv
import ast
import matplotlib.pyplot as plt
import math

from tqdm import tqdm

def get_freq():
    cs = pd.read_csv('customers.csv')
    cs = cs['customer_id'].to_numpy()
    # at = pd.read_csv('articles.csv\\articles.csv')
    # at = at['article_id'].to_numpy()
    at_dic = {}
    for c in cs:
        at_dic[c] = {}
    cs = None
    with open('transactions_train.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            try:
                at_dic[line[1]][line[2]] += 1
            except:
                try:
                    at_dic[line[1]][line[2]] = 1
                except:
                    pass

    with open('frequency.csv', 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        for at in at_dic:
            wr.writerow([at, list(at_dic[at].items())])
    return at_dic

def merge():
    with open('frequency.csv', 'r', encoding='utf-8') as f:
        at_dic = csv.reader(f)
        with open('merge.csv', 'w', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(['customer_id', 'article_id', 'count'])
            for a in at_dic:
                for d in eval(a[1]):
                    wr.writerow([a[0], d[0], d[1]])

def get_interaction():
    df = pd.read_csv("frequency.csv", header=None)
    df = df.iloc[:,1].to_numpy()
    
    ana = {}
    for d in df:
        d = ast.literal_eval(d)
        count = 0
        for i in d:
            count += i[1]

        try:
            ana[count] += 1
        except:
            ana[count] = 1

    keys = sorted(list(ana.keys()))#구매횟수
    values = [ana[x] for x in keys]#key만큼 구매한 고객수
    multi = len(str(max(keys)))-2
    plt.bar(keys, values)
    plt.xlim(-10, math.ceil(max(keys)/(10**multi))*10**multi)
    plt.ylim(0, 160000)
    plt.xlabel("Purchase Count")
    plt.ylabel("Customers Count")
    plt.show()
    print('구매를 가장 많이 한 고객의 구매 횟수:', max(keys), '고객수:', ana[max(keys)])
    print('가장 많은 고객이 구매한 횟수:', max(values))
    print('고객들의 평균 구매 횟수:', (sum([x*ana[x] for x in keys]))/sum(values))


def matrix():#안댐 행렬이 너무큼
    df = pd.read_csv('merge.csv')
    df = df.pivot_table('count', index='article_id', columns='customer_id')
    df.to_csv("matrix.csv")

def interaction2():
    df = pd.read_csv('customers.csv')
    print(df.isna().sum())

def contents_filtering():
    df = pd.read_csv("articles.csv")
    drop_list = ['prod_name', 'product_type_name', 'product_group_name', 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name', 'department_name', 'index_code', 'index_name', 'index_group_name', 'section_name', 'garment_group_name', 'detail_desc']

    df = df.drop(drop_list, axis=1)
    article_id = df['article_id'].to_numpy()
    a_dict = {}
    for id in article_id:
        a_dict[id] = 0
    
    tr = pd.read_csv("transactions_train.csv")
    tr = tr['article_id'].to_numpy()

    for id in tr:
        a_dict[id] += 1
    
    tr = pd.DataFrame(a_dict.items(), columns=['article_id', 'count'])
    df = pd.merge(df, tr, how='outer', on='article_id')

    df.to_csv('articles_with_count.csv')

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
        wr.writerow(['t_dat', 'weekly_sales_count', 'weekly_top_sales'])
        for w in tqdm(ws.items()):
            sales = sorted(w[1].items(), key=lambda x: -x[1])
            wr.writerow([w[0], len(sales), sales])
            
def get_daily_dict():
    df = pd.read_csv("transactions_train.csv").drop(['customer_id', 'price', 'sales_channel_id'], axis=1).to_numpy()  
    daily_dict = {}

    for d in df:
        try:
            daily_dict[d[0]][d[1]] += 1
        except:
            try:
                daily_dict[d[0]][d[1]] = 1
            except:
                daily_dict[d[0]] = {}
                daily_dict[d[0]][d[1]] = 1

    with open("daily_sales_dict.csv", 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['t_dat', 'daily_sales_count', 'article_count'])
        for d in tqdm(daily_dict.items()):
            sales = sorted(d[1].items(), key = lambda x: -x[1])
            wr.writerow([d[0], len(sales), sales])

def get_ldbw():#kaggle 0.0225 https://www.kaggle.com/byfone/h-m-trending-products-weekly/notebook
    df = pd.read_csv("transactions_train.csv", usecols=['t_dat', 'customer_id', 'article_id'], dtype={'article_id':str})
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    last_ts = df['t_dat'].max()
    #print(last_ts)#2020-09-22 00:00:00

    df['ldbw'] = df['t_dat'].progress_apply(lambda d: last_ts-(last_ts-d).floor('7D'))#transactions.csv에서 가장 마지막 날짜를 기준으로 7일 씩 그룹화
    df.to_csv("last_day_of_billing_week.csv", index=None)

def get_weekly():#kaggle 0.0225 https://www.kaggle.com/byfone/h-m-trending-products-weekly/notebook
    df = pd.read_csv("last_day_of_billing_week.csv", index_col='Unnamed: 0')

    last_ts = pd.to_datetime(df['t_dat']).max()
    weekly_sales = df.drop("customer_id", axis=1).groupby(['ldbw', 'article_id']).count()
    weekly_sales = weekly_sales.rename(columns={'t_dat': 'count'})
    weekly_sales.to_csv("weekly_sales.csv")
    df = df.join(weekly_sales, on=['ldbw', 'article_id'])

    weekly_sales = weekly_sales.reset_index().set_index('article_id')
    last_day = last_ts.strftime('%Y-%m-%d')

    df = df.join(weekly_sales.loc[weekly_sales['ldbw']==last_day, ['count']], on='article_id', rsuffix='_targ')
    df['count_targ'].fillna(0, inplace=True)
    del weekly_sales

    df['quotient'] = df['count_targ'] / df['count']
    df.to_csv("weekly_sales.csv")

def above_30():
    df = pd.read_csv('frequency.csv').to_numpy()
    with open("23to26.csv", 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['customer_id', 'article_id', 'count'])
        for d in df:
            sum = 0
            for c in eval(d[1]):
                sum+=c[1]
            
            if sum >= 23 and sum <= 26:
                for c in eval(d[1]):
                    wr.writerow([d[0], c[0], c[1]])
# above_30()
# with open("23to26.csv", 'r', encoding='utf-8') as f:
#     valid_list = []
#     valid_count = -1
#     while True:
#         article = f.readline()
#         if not article: break
#         else:
#             article = article.split(',')[1]
#             if article not in valid_list:
#                 valid_list.append(article)
#                 valid_count += 1

def group_by_age():
    import os
    df = pd.read_csv("customers.csv")

    for i in range(1, 10):  
        if not os.path.exists('{0}대'.format(i*10)):
            os.mkdir('{0}대'.format(i*10))
        
        age_df = df.loc[(df['age'] >= i*10) & (df['age'] < (i+1)*10), 'customer_id']
        age_df.to_csv('{0}대\\{0}대_그룹.csv'.format(i*10), index=None)

def after_7_1_transacionts():
    df = pd.read_csv("transactions_train.csv")
    df.drop(['price', 'sales_channel_id'], axis=1)
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    df = df.loc[df['t_dat'] >=pd.to_datetime('2020-07-01')]
    df = df.to_numpy()
    transaction_dict = {}
    for d in df:
        customer = d[1]
        article = d[2]

        if customer not in transaction_dict:
            transaction_dict[customer] = []
        
        transaction_dict[customer].append(article)
    with open("2020-07-01_transactions.csv", 'w', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['customer_id', 'articles_id'])
        for transaction in transaction_dict.items():
            for t in transaction[1]:
                wr.writerow([transaction[0], t])

def age_transactions():
    df = pd.read_csv("2020-07-01_transactions.csv")
    for i in range(1, 10):
        print(i)
        group_df = pd.read_csv("{0}대\\{0}대_그룹.csv".format(i*10))
        group_transaction = pd.merge(df, group_df, how='right', on='customer_id')
        group_transaction.to_csv("{0}대\\{0}대_transactions.csv".format(i*10), index=None)

from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("10대\\10대_transactions.csv")

df.drop(df[df['articles_id'].isnull()].index, inplace=True)
df['articles_id'] = df['articles_id'].astype(int)

df['count'] = 1
df = df.pivot_table('count', index='customer_id', columns='articles_id')

df = df.fillna(0)

df = cosine_similarity(df)
df = pd.DataFrame(df)
df.to_csv("10대\\10대matrix.csv")