import FinanceDataReader as fdr
import pandas_datareader as pdr
import investpy
from pykrx import stock
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
import ta
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBClassifier
from pdpbox.pdp import pdp_isolate, pdp_plot

today = dt.date.today()
dd_mm_yyyy = today.strftime('%d/%m/%Y')
yyyy_mm_dd = today.strftime('%Y%m%d')

'''

kospi = pdr.DataReader("^KS11", "yahoo", start=dt.datetime(2000, 1, 1), end=today)
kospi.to_csv('kospi/kospi.csv')
kosdaq = pdr.DataReader("^KQ11", "yahoo", start=dt.datetime(2000, 1, 1), end=today)
kosdaq.to_csv('kospi/kosdaq.csv')

# 매크로 지표
usdkrw = pdr.DataReader("KRW=X", "yahoo", start=dt.datetime(2000, 1, 1), end=today)
usdkrw.to_csv('kospi/usdkrw.csv')
wti = investpy.search_quotes(text='WTI', products=['commodities'], n_results=1)
wti = wti.retrieve_historical_data(from_date='01/01/2000', to_date=dd_mm_yyyy)
wti.to_csv('kospi/wti.csv')
effr = pdr.DataReader("DFF", "fred", start=dt.datetime(2000, 1, 1), end=today)
effr.to_csv('kospi/effr.csv')
bok_rate =
usM2 = pdr.DataReader("WM2NS", "fred", start=dt.datetime(2000, 1, 1), end=today)
usM2.to_csv('kospi/usM2.csv')
#exports =

# 시장 지표
vix = pdr.DataReader("^VIX", "yahoo", start=dt.datetime(2000, 1, 1), end=today)
vix.to_csv('kospi/vix.csv')
sp500 = pdr.DataReader("^GSPC", "yahoo", start=dt.datetime(2000, 1, 1), end=today)
sp500.to_csv('kospi/sp500.csv')
vkospi = investpy.get_index_historical_data(index='KOSPI Volatility',
                                            country='South Korea',
                                            from_date='01/01/2000',
                                            to_date=dd_mm_yyyy)
vkospi.to_csv('kospi/vkospi.csv')
trading_value_by_investor = stock.get_market_trading_value_by_date("20180101", "20220321", "KOSPI")
trading_value_by_investor.to_csv('kospi/trading_Value_by_investor.csv')
short_value = stock.get_shorting_investor_value_by_date("20100101", "20220321", "KOSPI")
short_value.to_csv('kospi/short_volume.csv')
#macd =
#stochastic =
#rsi =
#bollinger =
#volume =

# 이익 지표
per_pbr_div = stock.get_index_fundamental("20000101", yyyy_mm_dd, "1001")
per_pbr_div.to_csv('kospi/per_pbr_div.csv')

'''



# 타겟 변수 생성하기

kospi = pd.read_csv('kospi/kospi.csv')
kospi.set_index('Date', inplace=True)
kospi.drop(columns=['High', 'Low', 'Close', 'Volume'], inplace=True)

bol_h = ta.volatility.bollinger_hband(kospi['Adj Close'])
bol_l = ta.volatility.bollinger_lband(kospi['Adj Close'])
rsi = ta.momentum.rsi(kospi['Adj Close'])
stochrsi = ta.momentum.stochrsi(kospi['Adj Close'])
technical = pd.concat([bol_l, bol_h, rsi, stochrsi], axis=1)


hm_days = [5, 20, 60, 120] # 영업일 기준이므로 일주일, 한 달, 3개월, 6개월 수익률이라고 보면 됨.
def process_data_for_labels(df):

    for hm_day in hm_days:
        df['return_{}d'.format(hm_day)] = (df['Adj Close'].shift(-hm_day) - df['Open'].shift(+1)) / df['Open'].shift(+1) # i day만큼 shift되어 future data를 불러오게 됨.

    df.fillna(0, inplace=True)

    return df

kospi = process_data_for_labels(kospi)

def buy_sell_hold(x):
    requirement = 0.00
    if x > requirement:
        return 1
    elif x < -requirement:
        return 0

for hm_day in hm_days:
    kospi['target_{}d'.format(hm_day)] = kospi['return_{}d'.format(hm_day)].apply(buy_sell_hold)
    # print(kospi['target_{}d'.format(hm_day)].value_counts(normalize=True)) # baseline 모델


# feature 살펴보기

per_pbr_div = pd.read_csv('kospi/per_pbr_div.csv', encoding='utf-8')
per_pbr_div.set_index('날짜', inplace=True) # per, pbr은 2002년 4월 23일부터 데이터 존재.
per_pbr_div.drop(columns = ['종가', '등락률', '선행PER'], inplace=True)
per_pbr_div.columns = ['PER', 'PBR', 'DIV']

effr = pd.read_csv('kospi/effr.csv')
effr.set_index('DATE', inplace=True)

bok_rate = pd.read_csv('kospi/bok_rate.csv', encoding='utf-8', header=None)
bok_rate.rename(columns={0: 'Date', 1: 'bok_rate'}, inplace=True)
bok_rate.set_index('Date', inplace=True)
bok_rate.index = pd.to_datetime(bok_rate.index, format='%Y-%m-%d')

sp500 = pd.read_csv('kospi/sp500.csv')
sp500.set_index('Date', inplace=True)
sp500['overnight_sp500'] = (sp500['Adj Close'] - sp500['Adj Close'].shift(1)) / sp500['Adj Close'].shift(1)
overnight_sp500 = pd.DataFrame(sp500['overnight_sp500'].shift(1)) # 한국 시간 기준으로 1월 4일에 알 수 있는 것은 12월 31일 대비 1월 3일날 sp500이 얼마나 올랐는지이다.

usdkrw = pd.read_csv('kospi/usdkrw.csv', encoding='utf-8', header=None, thousands=',')
usdkrw.rename(columns={0: 'Date', 1: 'usdkrw'}, inplace=True)
usdkrw.set_index('Date', inplace=True)
usdkrw.index = pd.to_datetime(usdkrw.index, format='%Y-%m-%d')

vix = pd.read_csv('kospi/vix.csv')
vix.set_index('Date', inplace=True)
vix.rename(columns={'Adj Close':'vix'}, inplace=True)
vix.drop(columns=['High', 'Low', 'Open', 'Close', 'Volume'], inplace=True)

vkospi = pd.read_csv('kospi/vkospi.csv')# 2013년 8월 6일부터 존재
vkospi.set_index('Date', inplace=True)
vkospi.rename(columns={'Close':'vkospi'}, inplace=True)
vkospi.drop(columns=['High', 'Low', 'Open', 'Volume', 'Currency'], inplace=True)

wti = pd.read_csv('kospi/wti.csv')
wti.set_index('Date', inplace=True)
wti.rename(columns={'Change Pct':'wti'}, inplace=True)
wti.drop(columns=['High', 'Low', 'Open', 'Close', 'Volume'], inplace=True)


# main dataframe 만들기

main_df = pd.concat([kospi, per_pbr_div], axis=1, join='inner')
main_df = main_df.merge(effr, how='left', left_index=True, right_index=True)
main_df = main_df.merge(vix.shift(1), how='left', left_index=True, right_index=True)
main_df = main_df.merge(overnight_sp500, how='left', left_index=True, right_index=True)
main_df = main_df.merge(vkospi, how='left', left_index=True, right_index=True)
main_df = main_df.merge(wti.shift(1), how='left', left_index=True, right_index=True)
main_df = main_df.merge(technical, how='left', left_index=True, right_index=True)

main_df.index = pd.to_datetime(main_df.index)

main_df = main_df.merge(bok_rate, how = 'left', left_index=True, right_index=True)
main_df = main_df.merge(usdkrw, how = 'left', left_index=True, right_index=True)


def boll_h(x):
    if x == True:
        return -1
    else:
        return 0
def boll_l(x):
    if x == True:
        return 1
    else:
        return 0
def rsi(x):
    if x >= .7:
        return -1
    elif x <= .3:
        return 1
    else:
        return 0
def stochrsi(x):
    if x >= .8:
        return -1
    elif x <= .2:
        return 1
    else:
        return 0

main_df['stochrsi'] = main_df['stochrsi'].apply(stochrsi)
main_df['rsi'] = main_df['rsi'].apply(rsi)
main_df['lband'] = main_df['lband'] > main_df['Adj Close']
main_df['hband'] = main_df['hband'] < main_df['Adj Close']
main_df['hband'] = main_df['hband'].apply(boll_h)
main_df['lband'] = main_df['lband'].apply(boll_l)

main_df['technical'] = main_df['stochrsi'] + main_df['lband'] + main_df['hband'] + main_df['rsi']

main_df.drop(columns=['lband', 'hband', 'stochrsi', 'rsi', 'Open', 'Adj Close', 'return_5d', 'return_20d', 'return_60d', 'return_120d'], inplace=True)

main_df.fillna(method='pad', inplace=True)
main_df.to_csv('kospi/main_df.csv')


'''
plt.subplot(2,2,1);
sns.violinplot(x=main_df.target_5d, y=main_df.PER, palette="Set3")
plt.ylim(3, )

plt.subplot(2,2,2);
sns.violinplot(x=main_df.target_20d, y=main_df.PER, palette="Set3")
plt.ylim(3, )

plt.subplot(2,2,3);
sns.violinplot(x=main_df.target_60d, y=main_df.PER, palette="Set3")
plt.ylim(3, )

plt.subplot(2,2,4);
sns.violinplot(x=main_df.target_120d, y=main_df.PER, palette="Set3")
plt.ylim(3, )

plt.show()
'''

'''
plt.subplot(2,2,1);
sns.violinplot(x=main_df.target_20d, y=main_df.PER, palette="Set3")

plt.subplot(2,2,2);
sns.violinplot(x=main_df.target_20d, y=main_df.PBR, palette="Set3")

plt.subplot(2,2,3);
sns.violinplot(x=main_df.target_20d, y=main_df['배당수익률'], palette="Set3")

plt.show()
'''


# train/val/test split

features = ['PER', 'PBR', 'DIV', 'DFF', 'overnight_sp500', 'bok_rate', 'usdkrw', 'vix', 'wti', 'technical']

train = main_df.loc['2010-01-01':'2019-12-31']
test = main_df.loc['2020-01-01':'2021-12-20']
X_train = train[features]
X_test = test[features]
y5_train = train['target_5d']
y20_train = train['target_20d']
y60_train = train['target_60d']
y120_train = train['target_120d']
y5_test = test['target_5d']
y20_test = test['target_20d']
y60_test = test['target_60d']
y120_test = test['target_120d']

y60_baseline = [1] * len(y60_test)
print(accuracy_score(y60_test, y60_baseline))
print(train.isna().sum())
# 기준모델 정확도

y5_baseline = [1]*len(y5_test)
print('5일 기준모델 정확도:', accuracy_score(y5_test, y5_baseline))
y20_baseline = [1]*len(y20_test)
print('20일 기준모델 정확도:', accuracy_score(y20_test, y20_baseline))
y60_baseline = [1]*len(y60_test)
print('60일 기준모델 정확도:', accuracy_score(y60_test, y60_baseline))
y120_baseline = [0]*len(y120_test)
print('120일 기준모델 정확도:', accuracy_score(y120_test, y120_baseline), '\n')


# Logistic Regression model

def Logistic(X_train, y_train, X_val, y_val):
    logistic = make_pipeline(StandardScaler(),
                             LogisticRegression(random_state=2))
    logistic.fit(X_train, y_train)
    print('훈련 정확도:', logistic.score(X_train, y_train))
    print('검증 정확도:', logistic.score(X_val, y_val), '\n')

Logistic(X_train, y60_train, X_test, y60_test)



# decision tree model

def decisiontree(X_train, y_train, X_val, y_val):
    tree = make_pipeline(
        DecisionTreeClassifier(random_state=2)
    )

    tree.fit(X_train, y_train)
    print('훈련 정확도', tree.score(X_train, y_train))
    print('검증 정확도', tree.score(X_val, y_val), '\n')

    '''
    model = tree.named_steps['decisiontreeclassifier']
    importances = pd.Series(model.feature_importances_, X_train.columns)
    importances.sort_values().plot.barh();
    plt.figure(figsize=(5, 15))
    '''

decisiontree(X_train, y60_train, X_test, y60_test)



# random forest model

def randomforest(X_train, y_train, X_val, y_val):
    rf = make_pipeline(
        RandomForestClassifier(random_state=2)
    )

    rf.fit(X_train, y_train)
    print('훈련 정확도', rf.score(X_train, y_train))
    print('검증 정확도', rf.score(X_val, y_val), '\n')

    '''
    model = rf.named_steps['randomforestclassifier']
    importances = pd.Series(model.feature_importances_, X_train.columns)
    plt.figure(figsize=(5, 15))
    importances.sort_values().plot.barh();
    '''

randomforest(X_train, y60_train, X_test, y60_test)


# xgboost model

def xgboost(X_train, y_train, X_val, y_val):
    xgb = XGBClassifier(random_state=2)
    xgb.fit(X_train, y_train)
    print('훈련 정확도', xgb.score(X_train, y_train))
    print('검증 정확도', xgb.score(X_val, y_val), '\n')

xgboost(X_train, y60_train, X_test, y60_test)


# 파라미터 조정



model = DecisionTreeClassifier(random_state=2)
param_search = {'max_depth': range(30, 50),
                'min_samples_split': range(20, 40),
                'max_features' : ['auto', 'log']
                }
tscv = TimeSeriesSplit(n_splits=10)
randomizedCV = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_search,
        cv=tscv,
        scoring='accuracy',
        n_iter=500,
        n_jobs=-1,
    )
'''
randomizedCV.fit(X_train, y60_train)
print('decision tree best score:', randomizedCV.best_score_)
print('decision tree best model:', randomizedCV.best_estimator_)


best_dtmodel = randomizedCV.best_estimator_
y60_test_pred = best_dtmodel.predict(X_test)
print(classification_report(y60_test, y60_test_pred))
print('auc score:', roc_auc_score(y60_test, y60_test_pred))

importances = pd.Series(best_dtmodel.feature_importances_, X_train.columns)
plt.figure(figsize=(10, 8))
plt.title('Decision Tree: Feature Importances')
importances.sort_values().plot.barh();

isolated = pdp_isolate(
    model=best_dtmodel,
    dataset=X_test,
    model_features=X_test.columns,
    feature='usdkrw',
    grid_type='percentile', # default='percentile', or 'equal'
    num_grid_points=10 # default=10
    )
pdp_plot(isolated, feature_name='usdkrw');
plt.show()
'''

model = RandomForestClassifier(random_state=2)
param_search = {'max_depth': [10, 15, 20, 30, 40, 50],
                    'max_features': ['auto', 'sqrt'],
                    'min_samples_leaf': [1, 2, 4],
                    'min_samples_split': [2, 5, 10],
                    'n_estimators': [1000, 1200, 1400]}
tscv = TimeSeriesSplit(n_splits=10)
randomizedCV = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_search,
        cv=tscv,
        scoring='accuracy',
        n_iter=100,
        n_jobs=-1,
        verbose=10
    )
'''
randomizedCV.fit(X_train, y60_train)
print('랜덤포레스트 best score:', randomizedCV.best_score_)
print('랜덤포레스트 best model:', randomizedCV.best_estimator_)


best_rfmodel = randomizedCV.best_estimator_
y60_test_pred = best_rfmodel.predict(X_test)
print(classification_report(y60_test, y60_test_pred))
print('auc score:', roc_auc_score(y60_test, y60_test_pred) )

importances = pd.Series(best_rfmodel.feature_importances_, X_train.columns)
plt.figure(figsize=(10, 8))
importances.sort_values().plot.barh();

features = []
isolated = pdp_isolate(
    model=best_rfmodel,
    dataset=X_test,
    model_features=X_test.columns,
    feature='usdkrw',
    grid_type='percentile', # default='percentile', or 'equal'
    num_grid_points=10 # default=10
    )
pdp_plot(isolated, feature_name='usdkrw');
plt.show()
'''


xgb = XGBClassifier(use_label_encoder=False, random_state=2)
params = {
    'min_child_weight': [0.1, 1],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.0001, 0.001, 0.1, 1],
    'n_estimators': [50, 100, 250],
    'reg_alpha': [0.0001, 0.001],
    'reg_lambda': [0.0001, 0.001, 0.1]
    }

tscv = TimeSeriesSplit(n_splits=10)
randomizedCV = RandomizedSearchCV(xgb,
                                        cv=tscv,
                                        param_distributions=params,
                                        n_iter=50,
                                        scoring='precision',
                                        n_jobs=-1,
                                      verbose=10,
                                      random_state=2
                                          )

randomizedCV.fit(X_train, y60_train)
print('xgboost best score:', randomizedCV.best_score_)
print('xgboost best model:', randomizedCV.best_estimator_)


best_xgbmodel = randomizedCV.best_estimator_
y60_test_pred = best_xgbmodel.predict(X_test)
print(classification_report(y60_test, y60_test_pred))
print('auc score:', roc_auc_score(y60_test, y60_test_pred))

importances = pd.Series(best_xgbmodel.feature_importances_, X_train.columns)
plt.figure(figsize=(10, 8))
importances.sort_values().plot.barh();
plt.show()


# PDP Plot 그리기

features = ['DIV', 'DFF', 'PER', 'vix']
isolated = pdp_isolate(
        model=best_xgbmodel,
        dataset=X_test,
        model_features=X_test.columns,
        feature='DIV',
        grid_type='percentile', # default='percentile', or 'equal'
        num_grid_points=10 # default=10
    )
pdp_plot(isolated, feature_name='DIV');
plt.show()

isolated = pdp_isolate(
        model=best_xgbmodel,
        dataset=X_test,
        model_features=X_test.columns,
        feature='bok_rate',
        grid_type='percentile', # default='percentile', or 'equal'
        num_grid_points=10 # default=10
    )
pdp_plot(isolated, feature_name='bok_rate');
plt.show()

isolated = pdp_isolate(
        model=best_xgbmodel,
        dataset=X_test,
        model_features=X_test.columns,
        feature='vix',
        grid_type='percentile', # default='percentile', or 'equal'
        num_grid_points=10 # default=10
    )
pdp_plot(isolated, feature_name='vix');
plt.show()

isolated = pdp_isolate(
        model=best_xgbmodel,
        dataset=X_test,
        model_features=X_test.columns,
        feature='usdkrw',
        grid_type='percentile', # default='percentile', or 'equal'
        num_grid_points=10 # default=10
    )
pdp_plot(isolated, feature_name='usdkrw');
plt.show()

isolated = pdp_isolate(
        model=best_xgbmodel,
        dataset=X_test,
        model_features=X_test.columns,
        feature='PBR',
        grid_type='percentile', # default='percentile', or 'equal'
        num_grid_points=10 # default=10
    )
pdp_plot(isolated, feature_name='PBR');
plt.show()

isolated = pdp_isolate(
        model=best_xgbmodel,
        dataset=X_test,
        model_features=X_test.columns,
        feature='overnight_sp500',
        grid_type='percentile', # default='percentile', or 'equal'
        num_grid_points=10 # default=10
    )
pdp_plot(isolated, feature_name='overnight_sp500');
plt.show()






