import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import json
import requests
import matplotlib.dates as mdates
import datetime as dt
import itertools
from sklearn.cluster import KMeans




def testfunc():
    return True

def expand_months_manual(prod_list,trend):
    '''
    @description: this function expands the dataframe from the API call to include the columns for the   months and the
                prices of each of the products during that month
@params:
    - product_list: the output of the API call and key 'products'
    - trend: trend to search, i.e. "high waisted jeans"
@return: returns the expanded dataframe
    '''
    #first populate the non nested price columns
    num_item = len(prod_list) 
    retailer_l = []
    prod_id_l = []
    prod_name_l = []
    
    for item_dict in prod_list:    
        retailer_l.append(item_dict["retailer"]["name"])
        prod_id_l.append(item_dict["id"])
        prod_name_l.append(item_dict["name"])
        
        
    data = {'retailer': retailer_l, 'product id': prod_id_l,'product name': prod_name_l}        
    trend_df=pd.DataFrame.from_dict(data) 
    trend_df['trend'] = trend #repeat trend terms for all products     
    
    month_cols = ['2018-01','2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09',\
                 '2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06',\
                  '2019-07','2019-08','2019-09']

    for c in month_cols:
        trend_df[c] = np.nan
        
    for i in range(trend_df.shape[0]):
        item_dict = prod_list[i]
        date_price = [(item_dict['nestedPriceHistory'][j]['time'][:7],item_dict['nestedPriceHistory'][j]['price']) \
                              for j in range(len(item_dict['nestedPriceHistory']))]

        #for every row, fill out the price in each of the month columns
        for date,price in date_price:
            trend_df.at[i, date] = price
    
    return trend_df





def expand_agg_price(skuhist_list, trend_top, trend, retailer):
    '''
    @description: this function expands the dataframe from the API call to include the columns for the   months and the
                prices of each of the products during that month
    @params:
    - skuhis_list: the output of the API call and key 'products'
    - trend: trend to search, i.e. "high waisted jeans"
    - retailer: the retailer name
    - metric: count, median, min, max
    @return: returns the expanded dataframe, single row
    '''
    count_df= pd.DataFrame()
    count_df.at[0,"trend_top"] = trend_top
    count_df.at[0,"trend"] = trend
    count_df.at[0,"ret_id"] = retailer #only 1
    med_df= pd.DataFrame()
    med_df.at[0,"trend_top"] = trend_top
    med_df.at[0,"trend"] = trend
    med_df.at[0,"ret_id"] = retailer #only 1
    month_cols = ['2018-01','2018-02','2018-03','2018-04','2018-05','2018-06','2018-07','2018-08','2018-09',\
                 '2018-10','2018-11','2018-12','2019-01','2019-02','2019-03','2019-04','2019-05','2019-06',\
                  '2019-07','2019-08','2019-09']
    for c in month_cols:
        count_df[c] = np.nan
        med_df[c] = np.nan
        
    for b in skuhist_list:
        #if not b['price'][0] == {}:
        if not b['count'] == 0:    
            date = b['from'][:7]
            count = b['count']
            #med = b['price'][0]['usd']['median']
            med = b['price']['usd']['median']
            count_df.at[0,date] = count
            med_df.at[0,date] = med
    
    return count_df, med_df


def expand_agg_price_wk(skuhist_list, trend_top, trend, retailer):
    '''
    @description: this function expands the dataframe from the API call to include the columns for the   months and the
                prices of each of the products during that month
    @params:
    - skuhis_list: the output of the API call and key 'products'
    - trend: trend to search, i.e. "high waisted jeans"
    - retailer: the retailer name
    - metric: count, median, min, max
    @return: returns the expanded dataframe, single row
    '''
    count_df= pd.DataFrame()
    count_df.at[0,"trend_top"] = trend_top
    count_df.at[0,"trend"] = trend
    count_df.at[0,"ret_id"] = retailer #only 1
    med_df= pd.DataFrame()
    med_df.at[0,"trend_top"] = trend_top
    med_df.at[0,"trend"] = trend
    med_df.at[0,"ret_id"] = retailer #only 1
    '''
    for c in week_cols:
        count_df[c] = np.nan
        med_df[c] = np.nan
    '''    
    for b in skuhist_list:
        #if not b['price'][0] == {}:
        if not b['count'] == 0:    
            date = b['from'][:10]
            count = b['count']
            #med = b['price'][0]['usd']['median']
            med = b['price']['usd']['median']
            count_df.at[0,date] = count
            med_df.at[0,date] = med
    
    return count_df, med_df





### post flag t0
def findGreenSwitch(greenbarSeries):
    #skep the beginning NANs
    tempSeries = greenbarSeries.dropna()
    #print('finding switch, post dropna, start and end')
    #print(tempSeries.index[0], tempSeries[0],tempSeries.index[-1], tempSeries[-1])
    switchdatelist = []
    switchlist = []
    for i in range(1, len(tempSeries)):
        
        signlast = np.sign(tempSeries[i-1])
        signnow = np.sign(tempSeries[i])
        
        #consider 0 as positive
        if signlast == 1 or signlast == 0:
            if signnow == -1:
                switchdatelist.append(tempSeries.index[i])
                switchlist.append('pos_to_neg')
            else:#no change to signnow = 1 or 0
                pass
        elif signlast == -1:
            if signnow == 1 or signnow == 0:
                switchdatelist.append(tempSeries.index[i])
                switchlist.append('neg_to_pos')
        else: #in the rare case that signlast == 0
            pass #do nothing
        
    return switchdatelist, switchlist 



def findswitchind(switchdatelistdt, dateDT, way):
    '''
    @param:
        datestr is a string i.e. "2017-09-30"
        switchdatelistdt is a list of switchtimes in datetime format, it is sorted ascending
        way is 'post' or 'pre'
    @return 
        index of the first switch date that is either pre or post datestr
    '''
    #print('topDate {}'.format(dateDT))
    #print('switchdatelistdt')
    #print(switchdatelistdt)
    for i in range(len(switchdatelistdt)):
        #print('checking {}'.format(switchdatelistdt[i]))
        if switchdatelistdt[i]> dateDT:
            #print('found date > topDate')
            if way == 'pre':
                return i-1
            elif way == 'post':
                return i
            else: 
                #print('findswitchind param way is not pre or post')
                return -1 #catch all case
    #if the trend is continuouisily trending up, i.e. top Greenbar date is after all switches
    if way == 'pre':
        return len(switchdatelistdt)-1
    else: #way == 'post'
        return 0
    #return -1 

def check8wk(switchdatelistdt, startind, endind):
    
    if startind<0:
        return True
    if endind>len(switchdatelistdt)-1:
        return True
    
    #print("start {}, end {}".format(switchdatelistdt[startind],switchdatelistdt[endind]))
    weeks = (switchdatelistdt[endind] - switchdatelistdt[startind]).days/7
    #print("weeks between {}".format(weeks))
    return weeks >=8

def findt0(topdateStr, switchlist, switchdatelist, switchdatelistdt):
    '''
    function checks for the 'true start date' of a trend before peak greenbar date or topdateStr
    check for 8 weeks rule of short negative breaks
    @param
        topdateStr - peak greenbar date in string format
        switchlist - switch term list in string format
        switchdatelist - switchdatelist in string format
        switchdatedt = switchdatelist in datetime format
    
    @return t0 the start date of a trend in string
    '''
    topdateDT = dt.datetime.strptime(topdateStr, '%Y-%m-%d').date()
    ind = findswitchind(switchdatelistdt, topdateDT, 'pre')
    #print('ind post findswitchind {}'.format(ind))
    #print('len of switchdatelist {}'.format(len(switchdatelist)))
    if ind == -1:
        #print('no switch before topdate, set t0 to find_green')
        return 'first_green'
    #else:
    t0 = switchdatelist[ind] 
    #print("first ind {} first t0 {}".format(ind, t0))
    ttemp = t0 #placeholder
    state = 'pos'
    stop = False
    while not stop:
        #ind = findswitchind(switchdatelistdt, ttemp, 'pre')
        if ind <0: #we've reached the start of the datelist
            break
            
        ttemp = switchdatelist[ind]
        #print('new ttemp {}'.format(ttemp))
        
        if switchlist[ind] == "pos_to_neg":
            state = 'neg'
        else: state = 'pos'
        
        if state == 'neg':
            stop = check8wk(switchdatelistdt, ind, ind+1)#check if it is greater than 8 weeks
            
        else: #state == 'pos'
            #go to the next one before
            #print('pos state seting t0 to {}'.format(ttemp))
            t0 = ttemp
   
        ind-=1 #go to last switchdate    
    return t0
    #find the switch from neg_to_pos
    
    

def findtend(topdateStr, switchlist, switchdatelist, switchdatelistdt):
    '''
    function checks for the 'true end date' of a trend after peak greenbar date or topdateStr
    check for 8 weeks rule of short negative breaks
    @param
        topdateStr - peak greenbar date in string format
        switchlist - switch term list in string format
        switchdatelist - switchdatelist in string format
        switchdatedt = switchdatelist in datetime format
    
    @return tend the end date of a trend in string
    '''
    topdateDT = dt.datetime.strptime(topdateStr, '%Y-%m-%d').date()
    ind = findswitchind(switchdatelistdt, topdateDT, 'post')
    ttemp = switchdatelist[ind] 
    tend = ttemp
    #print("first ind {} first tend {}".format(ind, tend))
    state = 'neg'
    stop = False
    while not stop:
        
        if state == 'neg':
            #print('neg state')
            #print('change tend to {}'.format(ttemp))
            tend = ttemp
            stop = check8wk(switchdatelistdt, ind, ind+1)#check if it is greater than 8 weeks
            #print('stop is {}'.format(stop))
        else: #state == 'pos'
            #go to the next one before
            #print('pos state seting tend to {}'.format(ttemp))
            if ind ==len(switchdatelist)-1: #we'are at the end and it is just all potive:
                tend = '2019-08-25' #hard code to be the end of the time sries
            else:
                #tend = switchdatelist[ind+1]
                pass
            
        ind+=1 #go to next switch date
        if ind >len(switchdatelist)-1: #we've reached the end of the datelist
            break #end loop
        ttemp = switchdatelist[ind]
        
        if switchlist[ind] == "pos_to_neg":
            state = 'neg'
        else: state = 'pos'

    return tend
    
def find1greendate(greenbarSeries):
    '''
    return the date of the earliest positive green bar
    '''
    tempSeries = greenbarSeries.dropna() 
    result = tempSeries.index[0]
    for i in range(len(tempSeries)):
        if tempSeries[i] > 0: #as soon as a greenbar is positive
            result = tempSeries.index[i]
            break
    return result
       



def findTTT(trenddf, startdate):
    #tempdf has ['greenbar']
    #startdate is in format '2017-09-30'
    
    #first truncate trenddf by startdate
    dt_index = pd.to_datetime(trenddf.index)
    temp_df = trenddf[dt_index>startdate]
     
    
    greenbarS = temp_df['greenbar']
    switchdatelist, switchlist = findGreenSwitch(greenbarS) #string formats
    if len(switchdatelist) == 0: #there were no switches
        #print('switchdatelist is empty')
        #return ['2017-09-30', '2017-09-30', '2017-09-30']
        t0 = find1greendate(greenbarS)
        return [t0, t0, t0]

    switchdatelistdt = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in switchdatelist] 
    topgreenbar = sorted(temp_df['greenbar'].dropna())[-1]
    topdateStr = temp_df[temp_df['greenbar'] == topgreenbar].index[0] #str
    #print('topdateStr {}'.format(topdateStr))
    
    
    #find the switches before and after the topdate
    
      
    t0 = findt0(topdateStr, switchlist, switchdatelist, switchdatelistdt) #in string format
    if t0 == 'first_green': #when there are no switch dates before topdate
        #print('t0 {}, should be first positive green bar'.format(t0))
        t0 = find1greendate(greenbarS)
        #print('t0 is now {}'.format(t0))

    tend = findtend(topdateStr, switchlist, switchdatelist, switchdatelistdt) #in string format
    ttop = topdateStr#when the trend peaked

    return [t0, ttop, tend]

'''
returns t0 without plotting it
'''
def return_t0(term, search_data):
    print('calculating t0 for {}'.format(term))
    trend_df = pd.DataFrame(index = search_data.columns)
    trend_df[term] = search_data.loc[term] #this is a series
    trend_df["3wkMA"] = trend_df[term].rolling(window=3).mean()
    trend_df["12wkMA"] = trend_df[term].rolling(window=12).mean()
    trend_df['greenbar'] = trend_df["3wkMA"] - trend_df["12wkMA"]
    resultL = findTTT(trenddf=trend_df, startdate='2017-09-30')
    
    return resultL[0]
    
def plot_t0(term, search_data):
    print('calculating t0 for {}'.format(term))
    trend_df = pd.DataFrame(index = search_data.columns)
    trend_df[term] = search_data.loc[term] #this is a series
    trend_df["3wkMA"] = trend_df[term].rolling(window=3).mean()
    trend_df["12wkMA"] = trend_df[term].rolling(window=12).mean()
    trend_df['greenbar'] = trend_df["3wkMA"] - trend_df["12wkMA"]
    resultL = findTTT(trenddf=trend_df, startdate='2017-09-30')

    #plt.plot(list(search_data.columns), search_data.loc['neon'], color = 'b')
    x = trend_df.index
    y = trend_df[term]
    y2 = trend_df['3wkMA']
    y3 = trend_df['12wkMA']
    y4 = trend_df['greenbar']
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(x, y, color = 'b')
    ax.plot(x, y2, color = 'r')
    ax.plot(x, y3, color = 'c')
    ax.bar(x, y4, color = 'g')

    fig.autofmt_xdate()
    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    ax.grid(True)

    start, end = ax.get_xlim()
    #print(start, end)
    ax.xaxis.set_ticks(np.arange(start, end, 10))

    for xc in resultL:
        plt.axvline(x=xc, label='line at x = {}'.format(xc))

    plt.legend()
    plt.show()
    
    return resultL[0] #returns t0


def findStartVal(temp_Ser):
    nrow = len(temp_Ser)
    for i in range(nrow):
        if not temp_Ser[i] == 0: #if the value is not 0
            return temp_Ser[i]
        #in case all are 0
    return -1

def makeRollingIndex (trend_df, t0, trend):
    '''
    this function takes in trend_df which is index has date (months) and 6 retailer columns 
    the trend_df values can be count or median price
    it sets t0 count/median price to 100
    it calcuates the % of count/median price relative to t0 count post t0
    returns new df with that data
    input trend_df is the data for a given trend
    
    @param
        trend_df: data frame with index as dates, columns as each retailer, data as either count/median price
        t0: the t0 of that trend
        trend: the trend term i.e. 'neon'
    
    '''
    
    dt_index = pd.to_datetime(trend_df.index)
    temp_df = trend_df[dt_index>=t0]
    temp_df = temp_df.fillna(0)
    result_df = pd.DataFrame(index = temp_df.index)
    
    for c in temp_df.columns:
        startVal = findStartVal(temp_df[c])
        result_df[c] = temp_df[c]/startVal
    
    return result_df

def addSearchIndex (result_df, search_data, term, t0_term):
    '''
    adds the search index of the term to the result_df which starts from index t0_term
    '''
   
    search_term = search_data.loc[term]
    dt_index = pd.to_datetime(search_term.index)
    temp_search_term = search_term[dt_index>=t0_term]
    result_df.index = pd.to_datetime(result_df.index)
    temp_search_term.index = pd.to_datetime(temp_search_term.index)
    result_df[term+'_trend'] = temp_search_term

    return result_df

'''
@description: this function plots multicurves of the product stock for a user-defined trend term
@params: search_data - google search terms, read in from a pickle file
         market_data - df_count_all_* or df_med_all_*
         trend - the actual trend you want to plot multicurves for (adds '%20' in spaces where applicable)
         retailers - a list of retailers you're specifically looking at
         rolling_window - number of weeks we apply rolling to to calcualte max slope, usual it is 3
         makeplot - flag to produce plot 
         count_data - if True, flag that data is count, if False, flag that data is price (this is for printing)
@return: max_slopes - dataframe with the max slopes per retailer/trend (trend is normalized by range of y values) 
'''
def plot_multicurve(search_data, market_data, trend, retailers, rolling_window, week_cols,
                    makeplot = True, count_data = True):
    #get all t0 times for the trend we want and turn it into strings
     
    t0 = return_t0(trend, search_data)
    t0_str = str(pd.to_datetime(t0)- pd.offsets.MonthBegin(1))[0:10]
    
    #format trend to include '%20' in spaces where applicable
    trend_formatted = '%20'.join(trend.split(' '))
    
    #fetch only the rows in the market data that correspond to our specified retailers and trend
    trend_df = market_data[(market_data["Retail Site"].isin(retailers)) & (market_data["trend"] == trend_formatted)].copy()
    
    #extract only the retail site and weeks from the columns in preparation for transposing the dataframe
    #week_cols = list(trend_df.columns[1:-2])
    #week_cols.remove('Retailer ID')
    #week_cols.remove('Crawled?')
     
    trend_df = trend_df[["Retail Site"] + week_cols]
    
    #transpose the current trend dataframe
    trend_dfT = trend_df.T
    collist = list(trend_dfT.loc["Retail Site"].values)
    trend_dfT.columns = collist
    trend_dfT = trend_dfT.drop(["Retail Site"])
    
    dt_index = pd.to_datetime(trend_dfT.index)
    temp_df = trend_dfT[dt_index >= t0_str]
    temp_df = temp_df.fillna(0)
    
    #roll results
    result_df = makeRollingIndex(temp_df, t0_str, trend)
    result_df2 = addSearchIndex(result_df, search_data, trend, t0_str)
    result_df2 = result_df2.rolling(window=rolling_window).sum()
    
    #get ratio of ranges to normalize trend column by
    min_retailer_val = result_df2[result_df2.columns[:-1]].min().min()
    max_retailer_val = result_df2[result_df2.columns[:-1]].max().max()
    
    trend_col_name = result_df2.columns[-1]
    trend_col = result_df2[trend_col_name]
    min_trend_val = trend_col.min()
    max_trend_val = trend_col.max()
    
    retailer_diff = max_retailer_val - min_retailer_val
    trend_diff = max_trend_val - min_trend_val
    
    delta_ratio = int(trend_diff/retailer_diff)
    
    #make dataframe of max slopes to return
    df_diff = result_df2.diff()
    max_b = pd.DataFrame(np.max(df_diff))
    
    #adjust/normalize slope for trend column
    max_b.loc[trend_col_name] = max_b.loc[trend_col_name]/delta_ratio
    
    max_dates = pd.DataFrame(df_diff.idxmax())
    max_b.columns = ['Max Slope']
    max_dates.columns = ['Date']
    
    max_slopes = pd.concat([max_dates, max_b], axis=1)
    ##CP edit
    max_slopes["t0"] = t0 
         
    if makeplot:
        #plot multicurve
        fig = plt.figure(figsize=(20, 10))
        time = result_df2.index
        ax = fig.add_subplot(111)

        
                   
        for r in retailers:
            ax.plot(time, result_df2[r], '-', label = r)
            ##CP edit
            #calcualte t0 of retailers and plot CP edit
            r_t0 = plot_t0_ret(market_data, r, trend, t0, makeplot = False)
            plt.axvline(x=r_t0, label='{} t0 = {}'.format(r, r_t0)) 
            ##CP edit    
            max_slopes.at[r, 't0'] = r_t0 
        #plot the t0 of the trend
        plt.axvline(x=t0, color = 'r', label='{} t0 = {}'.format(trend, t0))
        
        ax2 = ax.twinx()
        ax2.plot(time, result_df2[trend+'_trend'], marker = '*', color ='r', label = trend+'_trend')
        ax.legend(loc=0)
        ax2.legend(loc=0)
        ax.grid()
        ax.set_xlabel("Date")
        if count_data:
            ax.set_ylabel(r"Index of Item Counts")
        else:
            ax.set_ylabel(r"Index of Item Price")
        ax2.set_ylabel(r"Index of Search")
        
        plt.legend(loc = 'best')
        plt.show()
    
    return max_slopes

### helper functions to make t_retailer_0 ###

def makeRollingIndexPad(cnt_df, t0, trend, padweeks = 11): 
    '''
    modify make Rolling Index to include the 11 weeks before to do 3 wk and 12 wk moving averages
    @param
        cnt_df - Transpose of single retailer inventor count for a given trend
        t0 - string of that trend's t0
        trend - 'term' of the trend i.e. 'neon'
        padweeks - number of weeks before t0 to include, this is for rolling averages, default is 11 (to start at 12th week)
    
    '''
    cnt_df = cnt_df.fillna(0)
    dt_index = pd.to_datetime(cnt_df.index)
    
    t0_11 = (pd.to_datetime(t0) - pd.Timedelta(weeks=11)).strftime('%Y-%m-%d')
    temp_df = cnt_df[dt_index>=t0_11]
    result_df = pd.DataFrame(index = temp_df.index)
    
    for c in temp_df.columns:
        startVal = findStartVal(cnt_df[dt_index>=t0][c]) #find the first value that is after t0, not with 12wk before t0
        #print("startVal {}".format(startVal))
        result_df[c] = temp_df[c]/startVal
    
    return result_df


def plot_t0_ret(market_data, retailer, trend, t0, makeplot = True):
    '''
    plot the t0, tpeak, tend lines with retailer market_data, a.k.a inventory action
    @param:
        market_data - either inventory or price data (e.g. df_cnt_all or df_med_all)
        retaier - name of the retaielr (e.g. 'Asos')
        trend - name of the trend (e.g. 'one shoulder')
        t0 = the t0 of the trend
        plot = if True then make plots
    @return t0 for this retailer
    '''
    #print('plot_t0_ret retailer {}'.format(retailer))
    trend = '%20'.join(trend.split(' '))
    
    ret_df = market_data[(market_data['trend'] == trend) & (market_data['Retail Site'] == retailer)]
    #week_cols = list(ret_df.columns[4:-2])
    week_cols = list(ret_df.columns[6:])
    ret_dfT = ret_df[week_cols].T
    ret_dfT.columns = ['count']
    #normalized inventory action as a ratio to t0 with 11 week padding before t0
    ret_norm_df = makeRollingIndexPad(ret_dfT, t0, trend, padweeks = 11) 
    
    #trend_df = pd.DataFrame(index = search_data.columns)
    #trend_df[term] = search_data.loc[term] #this is a series
    ret_norm_df["3wkMA"] = ret_norm_df['count'].rolling(window=3).mean()
    ret_norm_df["12wkMA"] = ret_norm_df['count'].rolling(window=12).mean()
    ret_norm_df['greenbar'] = ret_norm_df["3wkMA"] - ret_norm_df["12wkMA"]
    #change index format
    ret_norm_df.index = pd.to_datetime(ret_norm_df.index).strftime("%Y-%m-%d")
    resultL = findTTT(trenddf=ret_norm_df, startdate='2017-09-30')
    
    if makeplot:
        #plt.plot(list(search_data.columns), search_data.loc['neon'], color = 'b')
        x = ret_norm_df.index
        y = ret_norm_df['count']
        y2 = ret_norm_df['3wkMA']
        y3 = ret_norm_df['12wkMA']
        y4 = ret_norm_df['greenbar']
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(x, y, color = 'b', label = retailer)
        ax.plot(x, y2, color = 'r')
        ax.plot(x, y3, color = 'c')
        ax.bar(x, y4, color = 'g')

        fig.autofmt_xdate()
        ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

        ax.grid(True)

        start, end = ax.get_xlim()
        #print(start, end)
        ax.xaxis.set_ticks(np.arange(start, end, 10))

        for xc in resultL:
            plt.axvline(x=xc, label='line at x = {}'.format(xc))

        plt.legend()
        plt.show()
    
    return resultL[0] #returns t0


def getdelta_t0 (search_data, market_data, ret_list, trend, t0, week_cols):
    '''
    @return a dataframe that has the difference in t0 between retailers and trend
    '''
    result = pd.DataFrame()
    trend1W = '%20'.join(trend.split())
    #find the delta t0
    t0_rets = []
    for ret in ret_list:
        t0_ret = plot_t0_ret(market_data, ret, trend1W, t0, False)
        t0_rets.append(t0_ret)
        
    result["trend"] = [trend]*len(ret_list)
    result["trend_t0"] = t0
    result["retailer"] = ret_list
    result["ret_t0"] = t0_rets
    result["delta_t0_wk"] = (pd.to_datetime(t0_rets) - pd.to_datetime(t0)).days/7
    
    #find the max slope
    rolling_window = 3
    m_slope = plot_multicurve(search_data, market_data, trend, ret_list, rolling_window, week_cols, False, True) #supress plot
    m_slope = m_slope[['Date', 'Max Slope']]
    #also drop retailers with Max Slope == 0
    m_slope = m_slope[m_slope['Max Slope'] > 0] 
    trend_ms = m_slope.loc[trend+"_trend"]['Max Slope']
    #print(m_slope)
    #print('trend_ms {}'.format(trend_ms))
    m_slope["del_slope"] = m_slope['Max Slope'] - trend_ms
    m_slope.reset_index(inplace = True)
    m_slope.columns = ['retailer', 'm_slope_date', 'max_slope', 'del_slope']
    m_slope["trend_max_slope"] = trend_ms
    
    result = result.merge(m_slope, how="inner", left_on ="retailer", right_on = "retailer")
    
    #merge market index from market_data i.e. cnt_df_all
    ret_label = market_data.groupby(['Market Index', 'Retail Site']).count().reset_index()[['Market Index', 'Retail Site']]
    result = result.merge(ret_label, how = "inner", left_on = "retailer", right_on = "Retail Site")
    result = result[["trend", "trend_t0", "trend_max_slope","retailer", "Market Index","ret_t0", "delta_t0_wk","m_slope_date",
                    "max_slope", "del_slope" ]]
    #add del_log_slope
    result['del_log_slope'] = np.log(result['max_slope']) - np.log(trend_ms)
    
    return result

def val_retailers(market_data, trend):
    #return list of retailers where is is not all 0 and it is not 
    trend1W = '%20'.join(trend.split())
    #trend_df = market_data[market_data["trend"] == trend1W]
    retlist = []
    for ret in (market_data["Retail Site"].unique())[:-1]:
        #print("ret {}".format(ret))
        ret_row = market_data[(market_data["Retail Site"] == ret)&(market_data["trend"] == trend1W)]
        #ret_row = trend_df[trend_df["Retail Site"] == ret]
        #print(ret_row)
        #print(ret_row.shape)
        inc = True
        if ret_row.isnull().sum(axis = 1).values[0] == 87: #all rows is empty
            inc = False
        if ret_row.max(axis = 1).values[0] == ret_row.min(axis = 1).values[0]: #there is no fluctuation
            inc = False
        if ret_row.max(axis = 1).values[0] <=9: #not enough inventory    
            inc = False
        if inc:
            retlist.append(ret)
    
    return retlist

def market_data_fillna(week_cols):    
    '''
    saved filled df_cnt_all as df_cnt_all112019.csv
    '''
    market_data = pd.read_csv('df_cnt_all_111919.csv',index_col=0)
    market_data_ff = market_data[week_cols].T.fillna(method = 'ffill').fillna(0)
    MD2 = market_data_ff.T

    beg_cols = market_data.columns[:4]
    end_cols = market_data.columns[-2:]

    MD2[beg_cols] = market_data[beg_cols]
    MD2[end_cols] = market_data[end_cols]
    MD2 = MD2[list(beg_cols) + week_cols + list(end_cols)]

    MD2.to_csv("df_cnt_all_112019.csv")
    
def price_data_fillna(week_cols):    
    p_data = pd.read_csv('df_med_all_103119.csv',index_col=0)
    p_data_ff = p_data[week_cols].T.fillna(method = 'ffill').fillna(0)
    MD2 = p_data_ff.T
    beg_cols = p_data.columns[:4]
    end_cols = p_data.columns[-2:]
    MD2[beg_cols] = p_data[beg_cols]
    MD2[end_cols] = p_data[end_cols]
    MD2 = MD2[list(beg_cols) + list(week_cols) + list(end_cols)]
    MD2.to_csv("df_med_all112619.csv")
                    
    
    
'''
@description: this function plots a scatter plot for a specific trend's retailers stocking delay + trend 
@params: deltat0_df - the resulting dataframe from the getdelta_t0 function
         trend - trend to plot
         market_colors - a list of 5 colors that are used to plot the different market indices
@return: shows scatterplot for all retailers
'''
def plot_trend_scatterplot(deltat0_df,trend, plot_log = False, weight_cols = False):
    deltat0_df = deltat0_df.sort_values(by = 'Market Index')
    t0_wk_col = 'delta_t0_wk'
    slope_col = 'del_slope'
    log_slope_col = 'del_log_slope'
    
    if weight_cols:
        t0_wk_col = 'weighted_delta_t0_wk'
        slope_col = 'weighted_del_slope'
        log_slope_col = 'weighted_del_log_slope'
        
        
    x = deltat0_df[t0_wk_col]
    if plot_log:
        y = deltat0_df[log_slope_col]
    else:
        y = deltat0_df[slope_col]
    
    market_indices = deltat0_df['Market Index'].values
    market_colors = ['red','blue','green','cyan','magenta']
    retailers = deltat0_df['retailer'].values
    colors = itertools.cycle(market_colors)

    fig = plt.figure(figsize=(15,10))

    for i in range(len(x)):
        if i == 0:
            color = next(colors)
            plt.scatter(x.iloc[i], y.iloc[i], c=color, label=market_indices[i])
            
        #if this is a new market index label, switch colors
        if i != 0 and market_indices[i] != market_indices[i-1]:
            color = next(colors)
            plt.scatter(x.iloc[i], y.iloc[i], c=color, label=market_indices[i])
        else:
            #print
            plt.scatter(x.iloc[i], y.iloc[i], c=color)

        plt.annotate(retailers[i], (x.iloc[i], y.iloc[i]))

    plt.xlabel("Retailer Stocking Delay (weeks)")
    if plot_log:
        plt.ylabel("Retailer Stocking Aggression (log)")
    else:
        plt.ylabel("Retailer Stocking Aggression")
    #plt.title("Trend: {}".format(deltat0_df['trend'].unique()[0]))
    plt.title("Trend: {}".format(trend))
    plt.legend()
    plt.show()
    
    
'''
@description: this function takes all trends and returns a one large dataframe of all retailers' 
              combined trend and t0 delay information
              NOTE: some retailers don't have any inventory for a particular trend, so this function only
                    finds the retailers that are valid for that trend (i.e. number of retailers changes
                    per trend)
@params: search_data - google search trends
         market_data - df_count_all_...
         all_trends - all trends that you're interested in
@return: res_all - a combined dataframe of all the retailers and trends
'''
def combined_retailers_and_trends(search_data, market_data, all_trends, week_cols, makeplot=False):
    trends_t0 = [return_t0(t,search_data) for t in all_trends]
    res_i_list = []
    
    for i in range(len(all_trends)):    
        all_ret_i= val_retailers(market_data, all_trends[i]) #filter for all valid retailers
        res_i = getdelta_t0(search_data, market_data, all_ret_i, all_trends[i], trends_t0[i], week_cols)
        
        if makeplot: #plot the independent trend scatterplots per trend if you want
            plot_trend_scatterplot(res_i,trends[i])
            
        res_i_list.append(res_i)
        
    #concatenate all of the intermediate trend dataframes
    res_all = pd.concat(res_i_list, axis = 0)
    
    return res_all

'''
@description: creates a new dataframe with a weight column added that corresponds to the weight of a trend, i.e., the
              amount of that trend for a particular retailer divided by the total inventory for that retailer
@params: market_data - df_count....
         retailer - specific retailer
         week_cols - weeks we care about
@return: trend_weight_df - new dataframe with weight column
'''
def trend_weight_df(market_data, retailer, week_cols):
    #sum over the trends for every retailer
    market_data['inventory_sum'] = market_data[week_cols].sum(axis=1)
    trend_sum = market_data[['Retail Site','trend','inventory_sum']]
    #sum the above sums grouped by retailer
    retailer_groupby = market_data.groupby(['Retail Site']).agg({'inventory_sum':'sum'})
    
    trend_weight_df = trend_sum[trend_sum['Retail Site'] == retailer][['Retail Site','trend']]
    trend_weight_df['weight'] = trend_sum[trend_sum['Retail Site'] == retailer]['inventory_sum']/retailer_groupby.loc[retailer].values
    
    #format the trend column
    trends_with_spaces = []
    for s in list(trend_sum[trend_sum['Retail Site'] == 'Asos'].trend):
        trends_with_spaces.append(s.replace('%20',' '))
    trend_weight_df['trend'] = trends_with_spaces
    
    return trend_weight_df

'''
@description: this function takes the result of combined_retailers_and_trends ^^^ and returns a dataframe
              with the weighted mean weighted_delta_t0_wk and weighted_del_log_slope over all the trends 
              for a specified retailer
@params: combined_df - the result of combined_retailers_and_trends
         market_data - df_count_....
         retailer - the retailer in question
         week_cols - the weeks to include
@return: dataframe FOR ONE RETAILER that can be used for KMeans clustering after computing for all retailers
'''
def mean_all_trends_retailer(combined_df, market_data, retailer, week_cols):
    #new dataframe to hold everything
    output = pd.DataFrame(columns=['retailer','Market Index',
                                   'weighted_delta_t0_wk','weighted_del_log_slope'])
    #get retailer + trend combined info dataframe
    retailer_trends = combined_df[combined_df.retailer == retailer]  
    #get trend weights
    weight_df = trend_weight_df(market_data, retailer, week_cols)
    #join on the trend column
    join_df = pd.merge(retailer_trends, weight_df, on='trend')
    market_index = retailer_trends['Market Index'].unique()[0]
    
    #perform weighted average
    weight_delta_t0_wk = (join_df['delta_t0_wk']*join_df['weight']).mean()
    weight_del_log_slope = (join_df['del_log_slope']*join_df['weight']).mean()
    output.loc[0] = {'retailer':retailer,'Market Index':market_index,
                         'weighted_delta_t0_wk':weight_delta_t0_wk,
                         'weighted_del_log_slope':weight_del_log_slope}

    return output

'''
@description: applies weights to all rows in market_data where the market_data for a specific retailer is nonempty
@param: combined_df - the output from combined_retailers_and_trends
        market_data - df_count_....
        week_cols - weeks we care about
@return: res_all - concatenated dataframe with weighted weighted_delta_t0_wk and weighted weighted_del_log_slope to
         for all retailers
'''
def combined_weighted_data(combined_df, market_data, week_cols, custom_weight = False):
    if custom_weight:
        res_i_list = []

        for r in market_data['Retail Site'].unique():
            try:
                res_i = mean_all_trends_retailer(combined_df, market_data, r, week_cols)
                res_i_list.append(res_i)
            except:
                #print("except for retailer {}".format(r))
                continue

        #concatenate all of the intermediate trend dataframes
        res_all = pd.concat(res_i_list, axis = 0)

        return res_all
    
    else: #take average across all trends
        comb_dfLt = combined_df[['trend', 'Market Index', 'retailer', 'delta_t0_wk', 'del_log_slope']]
        output = comb_dfLt.groupby(['retailer','Market Index']).agg(\
                {"delta_t0_wk": 'mean','del_log_slope': 'mean'}).reset_index().rename(\
                columns = {"delta_t0_wk": 'weighted_delta_t0_wk', 
                           'del_log_slope': 'weighted_del_log_slope'})
        return output
                                          
def plotMultRetScatter(combined_df, retailers):
    '''
    functions plots the scatter plots of multiple retailers  across the trends
    max retailers should be 5
    @param: 
        combined_df: has delta_t0_wk, del_slope, del_log_slope columns for each retailer and each trend
        retailers: list of retilers, maxmium 5
    
    '''
    if len(retailers)>5:
        print('retailers cannot be more than 5!')
        return
    
    market_colors = ['red','blue','green','cyan','magenta']
    colors = itertools.cycle(market_colors)
    
    fig = plt.figure(figsize=(10,8))
    for r in retailers:
        r_df = combined_df[combined_df.retailer == r]
        x = r_df['delta_t0_wk'].values
        y = r_df['del_log_slope'].values
        trend_v = r_df['trend'].values
        plt.scatter(x, y, color = next(colors), label = r)
    
        for i in range(len(x)):
            plt.annotate(trend_v[i], (x[i], y[i]))
    plt.xlabel("Retailer Stocking Delay (weeks)")
    plt.ylabel("Retailer Stocking Aggression (log)")
    plt.title("Asos vs. Khols action across Trends")
    plt.legend(loc = 'best')
    plt.show()
    

def convertLabel(y_pred, centers):
    sorted_idx = np.argsort([np.linalg.norm(c) for c in centers])
    return y_pred.map({sorted_idx[0]:'early', sorted_idx[1]:'medium', sorted_idx[2]:'late'})    
    
def kMeansClustering(cluster_df, k, makeplot=False, weight_cols=True):
    if k!= 3:
        print("k should be set to 3")
        return
    
    t0_wk_col = 'delta_t0_wk'
    slope_col = 'del_slope'
    log_slope_col = 'del_log_slope'
    
    if weight_cols:
        t0_wk_col = 'weighted_delta_t0_wk'
        slope_col = 'weighted_del_slope'
        log_slope_col = 'weighted_del_log_slope'
    
    
    #add market index label
    market_indices = np.unique(cluster_df['Market Index'].values)
    market_index_labels = {market_indices[i]:i for i in range(len(market_indices))} #map
    X = cluster_df[[t0_wk_col,log_slope_col,'Market Index']].copy()
    X['Market Index Label'] = X['Market Index'].map(market_index_labels)
    
    #perform kmeans clustering
    kmeans = KMeans(n_clusters=k)
    x = X[t0_wk_col]
    y = X[log_slope_col]
    y_pred = kmeans.fit_predict(X[[t0_wk_col,log_slope_col]])
        
   #append predction to input cluster_df
    result_df = cluster_df.copy()
    result_df["Pred Num"] = y_pred
    centers = kmeans.cluster_centers_
    result_df["Pred Label"] = convertLabel(result_df["Pred Num"], centers)
    result_df = result_df.drop(columns = ['Pred Num'])
    
    if makeplot:
#         #plot actual clusters (do we actually need this??
#         #fig = plt.figure(figsize=(15,10))
#         plt.scatter(x,y,c=X['Market Index Label'])
#         plt.title("Actual Clusters")
#         #plt.legend()
#         plt.show()

        #plot retailer trend adoption strategies
        fig = plt.figure(figsize=(15,10))
        
        #speed_map = {'early':'fast','medium':'medium','late':'slow'}
        for speed, color in zip(['early','medium','late'],['mediumturquoise','gold','rebeccapurple']):
            speed_df = result_df[result_df['Pred Label'] == speed]
            x_speed = speed_df[t0_wk_col]
            y_speed = speed_df[log_slope_col]
            #plt.scatter(x_speed, y_speed,c = color, label = speed_map[speed])
            plt.scatter(x_speed, y_speed,c = color, label = speed)
            #label retailers in individual speed dataframe
            for i in range(speed_df.shape[0]):
                plt.annotate(speed_df['retailer'].values[i], (x_speed.values[i], y_speed.values[i]))
                
        
        plt.xlabel("Retailer Stocking Delay (weeks)")
        plt.ylabel("Retailer Stocking Aggression (log)")
        plt.title("Retailer Trend Adoption Strategies")
        plt.legend(loc='lower right')
        plt.show()
        
    return result_df, centers



def makeScoreCard(combined_df, agg_pred_df, trends):
    '''
        combined_df is cleaned of outliers
    '''
     
    res = agg_pred_df.rename(columns = {"Pred Label":"Pred Agg"})
    for trend in trends:
        temp = combined_df[combined_df["trend"] == trend]
        temp_result, _ = kMeansClustering(temp,3,makeplot=False, weight_cols = False)
        temp_result = temp_result.rename(columns = {"Pred Label":"Pred "+trend})
        res = res.merge(temp_result[["retailer", 'Pred '+trend]], 
                        how = 'left', left_on="retailer", right_on='retailer')

    res = res.fillna("X")
    
    return res