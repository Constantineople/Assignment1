# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:15:48 2023

@author: 123
"""

'''
为避免对复现策略实施干扰，idea的实现和结果运行统一放在此处
'''


import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
from matplotlib import pyplot as plt

connect_info = "sqlite:///FuturesMarketData.db"
engine = create_engine(connect_info)
sql_cmd = 'select * from AdjustedFuturesDaily'
df = pd.read_sql(sql=sql_cmd, con=engine, index_col='Instrument')

# conn=sqlite3.connect('FuturesMarketData.db')
# print('数据库打开成功')
# c=conn.cursor()#创建游标

# df = c.execute("select * from AdjustedFuturesDaily").fetchall()#按行来读，每一行都是一个tuple，合起来是个list
print(type(df))
# load_data_df_from_sql有两个接口，一个是品种列表，另一个是start_date
print(df.head())
instruments = df.index.unique()
print(instruments)


def load_data_df_from_sql(instruments, start_date):
    '''
    Parameters
    ----------
    instruments : list
        所需品种列表
    start_date : int(length:8)
        年月日如20180101
    Returns
    -------
    dict
    '''
    connect_info = "sqlite:///FuturesMarketData.db"
    engine = create_engine(connect_info)
    
    dict_data = {}
    
    df = pd.read_sql(sql=sql_cmd, con=engine, index_col='Instrument')
    df1 = pd.DataFrame(columns=df.columns)
    for i in instruments:
        df1 = df1.append(df.loc[i])#index是品种名
    df1.reset_index(inplace=True)#品种名单独一列，因为下面去除Trading Days< Start date的时候，要dropdf.loc[条件].index，如果index是品种会将同一品种的行全部删除
    df1.drop(df1.loc[df1['TradingDay'] < start_date].index, inplace=True)
    # 删除起始日前的数据，此项需放后面，否则可能因部分合约在start_date后不交易而报错
    df1['adjclose'] = df1['ClosePrice']*df1['factor_multiply']
    '''
    adjprice
    '''
    df2 = pd.DataFrame(
        {'TradingDay': df1['TradingDay'], 'Instrument': df1['index'], 'adjclose': df1['adjclose']}
        )
    #先设为双重索引，后利用unstack()使其与标准类似
    df2.reset_index(drop=True)
    df2 = df2.set_index('TradingDay')
    df2 = df2.set_index('Instrument', append=True)
    df2 = df2.unstack()
    #调整列名，此时列名是双重索引('adjclose',品种名)
    list1 = []
    for i in range(df2.shape[1]):
        list1.append(df2.columns[i][1])
    df2.columns = list1
    dict_data['adjclose'] = df2
    
    '''
    other data
    '''
    list2 = ['ClosePrice','HighestPrice','LowestPrice',
             'OpenInterest','OpenPrice','SettlementPrice',
             'TotalVolume','Turnover']
    for i in list2:
        dfi = pd.DataFrame({'TradingDay': df1['TradingDay'], 
                            'Instrument': df1['index'],
                            i:df1[i]})
        dfi.reset_index(drop=True)
        dfi = dfi.set_index('TradingDay')
        dfi = dfi.set_index('Instrument', append=True)
        dfi = dfi.unstack()
        listi = []
        for j in range(dfi.shape[1]):
            listi.append(dfi.columns[j][1])
        dfi.columns = listi
        dict_data[i] = dfi
    '''
    #point1 adjclose = (adjclose + highest + close)/3
    
    df_high = dict_data['HighestPrice']
    df_low = dict_data['LowestPrice']
    df_new_adjclose = (df_high+df_low+df2)/3#新的adjclose加入了对当日最高价和最低价的考量
    dict_data['adjclose'] = df_new_adjclose
    '''
    return dict_data


instruments = ['hc', 'rb', 'i', 'j', 'jm',
               'au', 'ag',
               'v', 'ru', 'l', 'pp', 'bu', 'TA', 'FG', 'MA',
               'y', 'p', 'm', 'a', 'c', 'cs', 'jd', 'RM', 'CF', 'SR', 'OI']
dict_data = {}
dict_data = load_data_df_from_sql(instruments, start_date=20180101)
dict_data['adjclose'].to_csv('adjclose(1).csv')

data = {}
data['price'] = dict_data['adjclose']
data['volume'] = dict_data['TotalVolume']
data['openinterest'] = dict_data['OpenInterest']

dict_parameter ={}
dict_parameter['window'] = 15   # lookback period for momentum
dict_parameter['trade_percent'] = 0.2  # long short percent
dict_parameter['hold_period'] = 2   # days to hold the position

#logreturns的数据接口一个是data，另一个是dict_parameter#即参数设置
def logreturns(data,dict_parameter):
    period = dict_parameter['window']
    df = data['price']
    volume = data['volume']
    #df1 = (df/volume).pct_change().rolling(period).mean()#pct_change(periods)计算与前periods位相差百分比
    '''
    #point3:每日价格/每日未平仓头寸的“收益率”作为动量因子
    openinterest = data['openinterest']
    df1 = (df.diff(1) / openinterest.diff(1)).pct_change().rolling(period).mean()
    '''
    #'''
    #point4:直接以每日价格/每日未平仓头寸作为因子
    openinterest = data['openinterest']
    df1 = (df.diff(1) / openinterest.diff(1)).rolling(period).mean()  # pct_change(periods)
    #'''
    return df1

signal = logreturns(data=data,dict_parameter=dict_parameter)
signal.to_csv('logreturns.csv')

#deltaneutral有三个参数,data,dict_parameter,以及signal(即因子表)
def deltaneutral(data,signal,dict_parameter):
    #获取long和short的df
    signal_rank = signal.rank(axis = 1)
    number = int(len(data['price'].columns)*dict_parameter['trade_percent'])#做多/做空组合的合约数
    long_max = number
    short_min = data['price'].shape[1] - number + 1
    position = signal_rank[(signal_rank<=long_max) | (signal_rank>=short_min)].fillna(0)
    #参数设置
    window = dict_parameter['window']#signal从0到window-1行是nan
    hold_period = dict_parameter['hold_period']#持有期H
    daily_fund = 100000#每个品种每日金额
    rules ={
            'IF':300,'IC':200,'IH':300,'TF':1000000,'T':1000000,
            'TS':2000000,'ag':15,'al':5,'au':1000,'bu':10,
            'cu':5,'fu':50,'hc':10,'ni':1,'pb':25,
            'rb':10,'ru':10,'sc':1000,'sn':1,'sp':10,
            'wr':10,'zn':5,'AP':10,'CF':5,'CJ':5,
            'CY':5,'FG':20,'JR':20,'LR':20,'MA':10,
            'OI':10,'PM':50,'RI':20,'RM':10,'RS':10,
            'SM':5,'SF':5,'SR':10,'TA':5,'WH':20,
            'ZC':100,'a':10,'b':10,'bb':500,'c':10,
            'cs':10,'eg':10,'fb':10,'i':100,'j':100,
            'jd':5,'jm':60,'l':5,'m':10,'p':10,
            'pp':5,'v':5,'y':10,'UR':20,'nr':10,
            'rr':10,'ss':5,'eb':5,'SA':20}#各品种交易1手单位数
    #交易首日设定
    firstday = position.iloc[window]
    firstday[(firstday>0)&(firstday<=long_max)] = 1/hold_period
    firstday[firstday>=short_min] = -1/hold_period
    position.iloc[window] = firstday
    danwei = []#用于记录所选每个instrument的1手单位数
    for i in position.columns:
        danwei.append(rules[i])
    #计算每日仓位(1/持有期为单位，暂不考虑资金等因素)，行列遍历
    length = position.shape[0] - window#自交易首日起有多少天
    for i in range(window+1,position.shape[0]):
        daily = position.iloc[i]
        daily[(daily>0)&(daily<=long_max)] = 1/hold_period
        daily[daily>=short_min] = -1/hold_period
        position.iloc[i] = daily#最新一日当日根据因子排名的持仓状况
        for j in range(position.shape[1]):
            if position.iloc[i-1][j]==0 :
                position.iloc[i][j] = position.iloc[i][j]
            elif position.iloc[i][j]!=0 and position.iloc[i-1][j]!=0:
                if (position.iloc[i][j]*position.iloc[i-1][j])>0:
                    #同号则相加，加后绝对值<=1则为此数，>1则改为1
                    agg = np.abs(position.iloc[i][j]) + np.abs(position.iloc[i-1][j])
                    if agg>1:
                        agg = 1
                    if position.iloc[i][j] >0:
                        symbol = 1
                    else:
                        symbol = -1
                    position.iloc[i][j] = symbol*agg
                    #异号则仓位直接是今天的数
            elif position.iloc[i][j]==0 and position.iloc[i-1][j]!=0:
                delta = np.abs(position.iloc[i-1][j]) - np.abs(position.iloc[i-2][j])
                if position.iloc[i-1][j]>0:
                    symbol = 1
                else:
                    symbol = -1
                if delta == 0 or delta<0:
                    today_value = np.abs(position.iloc[i-1][j]) - 1/hold_period
                    today_value = today_value*symbol
                    position.iloc[i][j] = today_value
                else:
                    position.iloc[i][j] = position.iloc[i-1][j]
    '''
        #获取每日持仓list
        signal_0 = signal.dropna()
        signal_T = signal_0.T#转置以便每日sort
        number = int(len(data['price'].columns)*dict_parameter['trade_percent'])#做多/做空组合的合约数
        dict_portfolio = {}
        for date in signal_0.index:
            dict_portfolio_daily = {}
            long_list = []
            short_list = []
            signal_T.sort_values(by = date,inplace = True,ascending = False)#降序排列：靠前的因子是因子值更大的
            long_list = list(signal_T.iloc[0:number].index)
            short_list = list(signal_T.iloc[-1*number:].index)
            dict_portfolio_daily['long'] = long_list
            dict_portfolio_daily['short'] = short_list
            dict_portfolio[date] = dict_portfolio_daily
        #持仓表
        period = dict_parameter['hold_period']#持有期，取下一个建仓日需加上period
        position_list_df = pd.DataFrame(dict_portfolio).T#转置后row是日期columns是long,short
        JianCangRi_index = list(range(0,len(position_list_df.index),period))
        JianCangRi = [position_list_df.index[x] for x in JianCangRi_index]
        df = pd.DataFrame({'long':[],'short':[]})
        for i in JianCangRi:
            df =df.append(position_list_df.loc[i])#列相同时，行可以直接append
            #df即为建仓日的position_list
    '''
        
    '''
        for i in range(0,len(JianCangRi_index)):
            if JianCangRi_index[i] + period >position_list_df.shape[0]:
                break
            else:
                position_list_df.iloc[JianCangRi_index[i]:JianCangRi_index[i]+period] = df.iloc[i]
        #获得了每一日的持仓品种表
    '''
    #计算考虑资金，1手交易单位数等因素的仓位计算              
    #由于data['price']的列和signal的是一样的，因此不用另行调整
    #注意计算真实仓位的时候要把hold_period即H乘回去，否则每日投入的金额就不是10万，而是10万乘i/H(i=0..H)
    return position* daily_fund*hold_period/danwei/data['price'] 

position= deltaneutral(data,signal,dict_parameter)
position.to_csv('position(1).csv')

#cal_bkt根据仓位计算策略每日盈亏，成交量，成交额等指标,有两个参数，一个是每日adjclose，一个是持仓position
def cal_bkt(adjclose, position):
    #每日盈亏，注意仓位是手数，但价格变动要乘上每手交易单位数
    rules ={
            'IF':300,'IC':200,'IH':300,'TF':1000000,'T':1000000,
            'TS':2000000,'ag':15,'al':5,'au':1000,'bu':10,
            'cu':5,'fu':50,'hc':10,'ni':1,'pb':25,
            'rb':10,'ru':10,'sc':1000,'sn':1,'sp':10,
            'wr':10,'zn':5,'AP':10,'CF':5,'CJ':5,
            'CY':5,'FG':20,'JR':20,'LR':20,'MA':10,
            'OI':10,'PM':50,'RI':20,'RM':10,'RS':10,
            'SM':5,'SF':5,'SR':10,'TA':5,'WH':20,
            'ZC':100,'a':10,'b':10,'bb':500,'c':10,
            'cs':10,'eg':10,'fb':10,'i':100,'j':100,
            'jd':5,'jm':60,'l':5,'m':10,'p':10,
            'pp':5,'v':5,'y':10,'UR':20,'nr':10,
            'rr':10,'ss':5,'eb':5,'SA':20}#各品种交易1手单位数
    danwei = []#用于记录所选每个instrument的1手单位数
    for i in position.columns:
        danwei.append(rules[i])
    pnl = position.shift(axis=0, periods=1).fillna(0) * adjclose.diff(axis=0, periods=1)*danwei
    pnl.to_csv('pnl(1).csv')
    #每日成交量
    volume = np.abs(position).diff(axis = 0,periods = 1).fillna(0)
    #成交额
    turnover = volume*adjclose*danwei
    bkt_result ={'pnl':pnl,'volume':volume,'turnover':turnover,'pnl_ptf':pnl.sum(axis=1)}
    return bkt_result

bkt_result = cal_bkt(dict_data['adjclose'],position)

#cal_perf根据回测结果计算策略评价指标，有1个参数bkt_result
def cal_perf(bkt_result):
    #sharpe ratio = 16 * mean(portfolio daily pnl) / std(portfolio daily pnl),16约为sqrt(250)
    pnl = bkt_result['pnl']
    daily_pnl_mean = pnl.sum(axis = 1).mean()
    daily_pnl_std = pnl.sum(axis = 1).std()
    sharpe_ratio = 16*daily_pnl_mean/daily_pnl_std
    # pot = sum(portfolio daily pnl) / sum(portfolio daily turnovers) * 10000
    turnover = bkt_result['turnover']
    pot = (pnl.sum())/(turnover.sum().sum()) * 10000
    
    performance ={}
    performance['sharpe_ratio'] = sharpe_ratio
    performance['pot'] = pot
    return performance

performance = cal_perf(bkt_result)
print(performance)
plt.plot(bkt_result['pnl_ptf'].values.cumsum())
plt.show()
