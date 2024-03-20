import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import copy
import gc
import json
from pathlib import Path
import sys
import os
import pickle
import joblib
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime, date, time
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold
import pytorch_lightning as pl
from hyperopt import hp
import numpy as np
import lightgbm as lgb
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, anneal, rand
from hyperopt import space_eval
from functools import partial
import warnings
warnings.filterwarnings("ignore")



pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin

def minMaxScaler(df):
    for column in df.columns:
    #   print(column)
      if column =='race_date' or column =='trainer' or column =='horse'or column =='jockey'or column == 'round_detail' or column == 'id' or column =='pre_jockey_race_date' or column=='pre_trainer_race_date' or column =="target":
        pass
      else:
        # 각 변수의 최소값 및 최대값 찾기
        min_val = df[column].min()
        max_val = df[column].max()
        # min-max 스케일링 적용
        df[column] = (df[column] - min_val) / (max_val - min_val)
    df.fillna(0, inplace=True)
    return df

def getJockey(start_table):
      ## jockey 이름을 고유 번호로 변경
    jockeys = start_table['jockey'].unique().tolist()
    jockey_ids = pd.DataFrame(index=jockeys, data={'jockey_id': range(1, len(jockeys)+1)}).to_dict()['jockey_id']
    start_table['jockey_id'] = start_table['jockey'].map(jockey_ids)

    # 조키별 경주 참여 횟수, 우승 횟수, 상위 3위 내 들어온 횟수 계산
    cnt = start_table[['jockey', 'race_date', 'position_at_finish']].groupby(['jockey', 'race_date']).count()
    # 각 경주에 조키가 참여한 횟수
    cnt.columns=['cnt']

    wins = start_table.loc[start_table['position_at_finish']<2, 
                        ['jockey', 'race_date', 'position_at_finish']].groupby(['jockey', 'race_date']).count()
    # 조키가 우승한 횟수
    wins.columns=['win_cnt']

    places = start_table.loc[start_table['position_at_finish']<4, 
                        ['jockey', 'race_date', 'position_at_finish']].groupby(['jockey', 'race_date']).count()
     # 조키가 상위 3위 안에 든 횟수
    places.columns=['place_cnt']

    # 데이터프레임 인덱스 리셋 및 병합
    cnt = cnt.reset_index()
    wins = wins.reset_index()
    places = places.reset_index()


    cnt.sort_values(by=['jockey', 'race_date'], ascending=[True]*2, inplace=True)
    wins.sort_values(by=['jockey', 'race_date'], ascending=[True]*2, inplace=True)
    places.sort_values(by=['jockey', 'race_date'], ascending=[True]*2, inplace=True)

    cnt = cnt.merge(wins, on=['jockey', 'race_date'], how='left')
    cnt = cnt.merge(places, on=['jockey', 'race_date'], how='left')
    cnt.fillna(0, inplace=True)

    # 누적 합계(cumsum) 및 비율 계산
    cumsum = cnt.groupby('jockey').cumsum()
    cnt['cnt_cumsum'] = cumsum['cnt']
    cnt['win_rate'] = cumsum['win_cnt']/cumsum['cnt']
    cnt['place_rate'] = cumsum['place_cnt']/cumsum['cnt']

    # 이전 경주 데이터로 시프트하여 새 컬럼 생성
    cnt_ = cnt.groupby('jockey').shift(1)

    # 시프트된 데이터를 새 컬럼으로 추가
    for column in cnt_.columns:
        cnt[f'pre_jockey_{column}'] = cnt_[column]
        
    cnt.fillna(0, inplace=True)

    # 계산된 통계 정보를 원본 start_table에 병합
    start_table = start_table.merge(cnt, on=['jockey', 'race_date'], how='left')
    # 우승 및 상위 3위 정보를 바이너리 값으로 변환
    # 우승 여부
    start_table['win'] = (start_table['position_at_finish']<2).astype(int)
    # 상위 3위 안에 들었는지 여부
    start_table['place'] = (start_table['position_at_finish']<4).astype(int)




    # 조키별, 경주별 데이터 분석을 위한 피벗 테이블 생성 및 데이터 전처리
    jockeys = start_table['jockey'].value_counts()[start_table['jockey'].value_counts()>=10].index.values.tolist()

    # 고유한 경주 날짜를 추출하고 정렬합니다.
    race_days = start_table['race_date'].unique().tolist()
    race_days.sort()

    # 경주 데이터로부터 조키별 승리 및 상위 입상 횟수를 요약하는 피벗 테이블을 생성합니다.
    data = pd.pivot_table(start_table, index=['race_date'], columns=['jockey'], 
                values=['win', 'place'], 
                aggfunc={'win': [np.sum, np.size], 'place': np.sum} )
    data.sort_index(ascending=True, inplace=True)

    dfs_pre = []
    # 조키 리스트를 순회하며 각 조키의 데이터를 처리합니다.
    for jk in jockeys:
        # 필요한 데이터를 선택하고, 누락된 값을 0으로 채웁니다.
        data_i = data[[('place', 'sum',            jk), 
                    (  'win', 'size',            jk), 
                    (  'win', 'sum',            jk)]].fillna(0)
        data_i.columns=['place', 'run', 'win']
        # 지난 31일 동안의 입상, 출전, 승리 횟수를 누적 합으로 계산합니다.

        data_i['plc_sum']=data_i['place'].rolling(11).apply(lambda x: x[:-1].sum())
        data_i['win_sum']=data_i['win'].rolling(11).apply(lambda x: x[:-1].sum())
        data_i['run_sum']=data_i['run'].rolling(11).apply(lambda x: x[:-1].sum())
         # 지난 31일 동안의 입상, 출전, 승리 횟수를 누적 합으로 계산합니다.
        data_i['plc_rate'] = data_i['plc_sum']/data_i['run_sum']
        data_i['win_rate'] = data_i['win_sum']/data_i['run_sum']
        data_i['jockey']=jk
        # 지난 3, 5, 10일 동안의 입상률과 승률의 변화를 계산합니다.
        for col in [ 'plc_rate', 'win_rate']:
            for i in [2,3,4,5,6,7,8,9,10]:
                data_i[f'{col}_diff{i}'] = data_i[col].diff(i)
        #dfs_pre.append(data_i.reset_index()[['jockey', 'race_date', 'win_sum', 'plc_sum', 'run_sum', 'plc_rate', 'win_rate']])
        dfs_pre.append(data_i.reset_index())
        
    # 조키별 데이터 프레임을 하나로 병합합니다.
    df_pre = pd.concat(dfs_pre)
    # 새로운 컬럼 이름을 설정합니다.

    new_cols = []
    for col in df_pre.columns:
        if col in ['jockey', 'race_date']:
            new_cols.append(col)
        else:
            new_cols.append(f'pre11_jockey_{col}')
        
    # print(new_cols)
    df_pre.columns = new_cols


    # 병합된 데이터를 기존의 `start_table` 데이터프레임에 병합합니다.
    start_table = start_table.merge(df_pre, on=['jockey', 'race_date'], how='left')
    # 병합된 데이터를 기존의 `start_table` 데이터프레임에 병합합니다.
    start_table.replace([-np.inf, np.inf], np.nan, inplace=True)

    start_table.isna().sum()[start_table.isna().sum()>0]

    start_table.fillna(0, inplace=True)
    start_table.isna().sum()[start_table.isna().sum()>0]

    return start_table



def getTrainer(start_table):
      ## jockey 이름을 고유 번호로 변경
    trainers = start_table['trainer'].unique().tolist()
    trainer_ids = pd.DataFrame(index=trainers, data={'trainer_id': range(1, len(trainers)+1)}).to_dict()['trainer_id']
    start_table['trainer_id'] = start_table['trainer'].map(trainer_ids)

    # 조키별 경주 참여 횟수, 우승 횟수, 상위 3위 내 들어온 횟수 계산
    cnt = start_table[['trainer', 'race_date', 'position_at_finish']].groupby(['trainer', 'race_date']).count()
    # 각 경주에 조키가 참여한 횟수
    cnt.columns=['cnt']

    wins = start_table.loc[start_table['position_at_finish']<2, 
                        ['trainer', 'race_date', 'position_at_finish']].groupby(['trainer', 'race_date']).count()
    # 조키가 우승한 횟수
    wins.columns=['win_cnt']

    places = start_table.loc[start_table['position_at_finish']<4, 
                        ['trainer', 'race_date', 'position_at_finish']].groupby(['trainer', 'race_date']).count()
     # 조키가 상위 3위 안에 든 횟수
    places.columns=['place_cnt']

    # 데이터프레임 인덱스 리셋 및 병합
    cnt = cnt.reset_index()
    wins = wins.reset_index()
    places = places.reset_index()


    cnt.sort_values(by=['trainer', 'race_date'], ascending=[True]*2, inplace=True)
    wins.sort_values(by=['trainer', 'race_date'], ascending=[True]*2, inplace=True)
    places.sort_values(by=['trainer', 'race_date'], ascending=[True]*2, inplace=True)

    cnt = cnt.merge(wins, on=['trainer', 'race_date'], how='left')
    cnt = cnt.merge(places, on=['trainer', 'race_date'], how='left')
    cnt.fillna(0, inplace=True)

    # 누적 합계(cumsum) 및 비율 계산
    cumsum = cnt.groupby('trainer').cumsum()
    cnt['cnt_cumsum'] = cumsum['cnt']
    cnt['win_rate'] = cumsum['win_cnt']/cumsum['cnt']
    cnt['place_rate'] = cumsum['place_cnt']/cumsum['cnt']

    # 이전 경주 데이터로 시프트하여 새 컬럼 생성
    cnt_ = cnt.groupby('trainer').shift(1)

    # 시프트된 데이터를 새 컬럼으로 추가
    for column in cnt_.columns:
        cnt[f'pre_trainer_{column}'] = cnt_[column]
        
    cnt.fillna(0, inplace=True)

    # 계산된 통계 정보를 원본 start_table에 병합
    start_table = start_table.merge(cnt, on=['trainer', 'race_date'], how='left')
    # 우승 및 상위 3위 정보를 바이너리 값으로 변환
    # 우승 여부
    start_table['win'] = (start_table['position_at_finish']<2).astype(int)
    # 상위 3위 안에 들었는지 여부
    start_table['place'] = (start_table['position_at_finish']<4).astype(int)




    # 조키별, 경주별 데이터 분석을 위한 피벗 테이블 생성 및 데이터 전처리
    trainers = start_table['trainer'].value_counts()[start_table['trainer'].value_counts()>=10].index.values.tolist()

    # 고유한 경주 날짜를 추출하고 정렬합니다.
    race_days = start_table['race_date'].unique().tolist()
    race_days.sort()

    # 경주 데이터로부터 조키별 승리 및 상위 입상 횟수를 요약하는 피벗 테이블을 생성합니다.
    data = pd.pivot_table(start_table, index=['race_date'], columns=['trainer'], 
                values=['win', 'place'], 
                aggfunc={'win': [np.sum, np.size], 'place': np.sum} )
    data.sort_index(ascending=True, inplace=True)

    dfs_pre = []
    # 조키 리스트를 순회하며 각 조키의 데이터를 처리합니다.
    for jk in trainers:
        # 필요한 데이터를 선택하고, 누락된 값을 0으로 채웁니다.
        data_i = data[[('place', 'sum',            jk), 
                    (  'win', 'size',            jk), 
                    (  'win', 'sum',            jk)]].fillna(0)
        data_i.columns=['place', 'run', 'win']
        # 지난 31일 동안의 입상, 출전, 승리 횟수를 누적 합으로 계산합니다.

        data_i['plc_sum']=data_i['place'].rolling(11).apply(lambda x: x[:-1].sum())
        data_i['win_sum']=data_i['win'].rolling(11).apply(lambda x: x[:-1].sum())
        data_i['run_sum']=data_i['run'].rolling(11).apply(lambda x: x[:-1].sum())
         # 지난 31일 동안의 입상, 출전, 승리 횟수를 누적 합으로 계산합니다.
        data_i['plc_rate'] = data_i['plc_sum']/data_i['run_sum']
        data_i['win_rate'] = data_i['win_sum']/data_i['run_sum']
        data_i['trainer']=jk
        # 지난 3, 5, 10일 동안의 입상률과 승률의 변화를 계산합니다.
        for col in [ 'plc_rate', 'win_rate']:
            for i in [2,3,4,5,6,7,8,9,10]:
                data_i[f'{col}_diff{i}'] = data_i[col].diff(i)
        #dfs_pre.append(data_i.reset_index()[['trainer', 'race_date', 'win_sum', 'plc_sum', 'run_sum', 'plc_rate', 'win_rate']])
        dfs_pre.append(data_i.reset_index())
        
    # 조키별 데이터 프레임을 하나로 병합합니다.
    df_pre = pd.concat(dfs_pre)
    # 새로운 컬럼 이름을 설정합니다.

    new_cols = []
    for col in df_pre.columns:
        if col in ['trainer', 'race_date']:
            new_cols.append(col)
        else:
            new_cols.append(f'pre11_trainer_{col}')
        
    # print(new_cols)
    df_pre.columns = new_cols


    # 병합된 데이터를 기존의 `start_table` 데이터프레임에 병합합니다.
    start_table = start_table.merge(df_pre, on=['trainer', 'race_date'], how='left')
    # 병합된 데이터를 기존의 `start_table` 데이터프레임에 병합합니다.
    start_table.replace([-np.inf, np.inf], np.nan, inplace=True)

    start_table.isna().sum()[start_table.isna().sum()>0]

    start_table.fillna(0, inplace=True)
    start_table.isna().sum()[start_table.isna().sum()>0]

    return start_table

def getHorse(start_table):
    
    horses = start_table['horse'].unique().tolist()
    horse_ids = pd.DataFrame(index=horses, data={'horse_id': range(1, len(horses)+1)}).to_dict()['horse_id']
    start_table['horse_id'] = start_table['horse'].map(horse_ids)
    ## ---------------------------------------------------------------------------------------------------------------------

    
    # # 각 말(horse_id)이 참여한 경주의 횟수를 계산합니다. 이는 말이 경주에 참여한 전체 횟수를 나타냅니다.
    cnt = start_table[['horse_id', 'race_date', 'position_at_finish']].groupby(['horse_id', 'race_date']).count()
    cnt.columns=['horse_cnt']
    # 각 말이 우승한 횟수를 계산합니다. 'position_at_finish'가 1인 경우를 우승으로 간주합니다.
    wins = start_table.loc[start_table['position_at_finish']<2, 
                        ['horse_id', 'race_date', 'position_at_finish']].groupby(['horse_id', 'race_date']).count()
    wins.columns=['horse_win_cnt']
    # 각 말이 상위 3위 안에 든 횟수를 계산합니다. 'position_at_finish'가 3 이하인 경우를 상위 3위 안에 든 것으로 간주합니다.

    places = start_table.loc[start_table['position_at_finish']<4, 
                        ['horse_id', 'race_date', 'position_at_finish']].groupby(['horse_id', 'race_date']).count()
    places.columns=['horse_place_cnt']

    # 위에서 계산한 결과를 병합하여 하나의 데이터프레임으로 만듭니다.
    cnt = cnt.reset_index()
    wins = wins.reset_index()
    places = places.reset_index()


    cnt.sort_values(by=['horse_id', 'race_date'], ascending=[True]*2, inplace=True)
    wins.sort_values(by=['horse_id', 'race_date'], ascending=[True]*2, inplace=True)
    places.sort_values(by=['horse_id', 'race_date'], ascending=[True]*2, inplace=True)

    cnt = cnt.merge(wins, on=['horse_id', 'race_date'], how='left')
    cnt = cnt.merge(places, on=['horse_id', 'race_date'], how='left')
    cnt.fillna(0, inplace=True)


    # 누적 합(cumsum)을 계산하여 말의 과거 성적을 나타내는 누적 지표를 만듭니다.
    cumsum = cnt.groupby('horse_id').cumsum()
    # 누적 참여 횟수, 누적 우승 횟수, 상위 3위 안에 든 누적 횟수를 새로운 컬럼으로 추가합니다.
    cnt['horse_cnt_cumsum'] = cumsum['horse_cnt']
    cnt['horse_win_rate'] = cumsum['horse_win_cnt']/cumsum['horse_cnt']
    cnt['horse_place_rate'] = cumsum['horse_place_cnt']/cumsum['horse_cnt']
    # 이전 경주의 데이터를 위해 데이터프레임을 시프트(shift)합니다. 이는 각 말의 최신 경주 결과를 제외한 과거 성적을 나타냅니다.
    cnt_ = cnt.groupby('horse_id').shift(1)
    # 위에서 시프트한 데이터프레임을 사용해 'pre_' 접두사가 붙은 새로운 컬럼들을 추가합니다.
    for column in cnt_.columns:
        cnt[f'pre_{column}'] = cnt_[column]
        
    cnt.fillna(0, inplace=True)
    #-------
    # 계산된 통계 정보를 원본 start_table에 병합
    start_table = start_table.merge(cnt, on=['horse_id', 'race_date'], how='left')
    # 우승 및 상위 3위 정보를 바이너리 값으로 변환
    # 우승 여부
    start_table['win'] = (start_table['position_at_finish']<2).astype(int)
    # 상위 3위 안에 들었는지 여부
    start_table['place'] = (start_table['position_at_finish']<4).astype(int)




    # 조키별, 경주별 데이터 분석을 위한 피벗 테이블 생성 및 데이터 전처리
    horse_ids = start_table['horse_id'].value_counts()[start_table['horse_id'].value_counts()>=10].index.values.tolist()

    # 고유한 경주 날짜를 추출하고 정렬합니다.
    race_days = start_table['race_date'].unique().tolist()
    race_days.sort()

    # 경주 데이터로부터 조키별 승리 및 상위 입상 횟수를 요약하는 피벗 테이블을 생성합니다.
    data = pd.pivot_table(start_table, index=['race_date'], columns=['horse_id'], 
                values=['win', 'place'], 
                aggfunc={'win': [np.sum, np.size], 'place': np.sum} )
    data.sort_index(ascending=True, inplace=True)

    dfs_pre = []
    # 조키 리스트를 순회하며 각 조키의 데이터를 처리합니다.
    for jk in horse_ids:
        # 필요한 데이터를 선택하고, 누락된 값을 0으로 채웁니다.
        data_i = data[[('place', 'sum',            jk), 
                    (  'win', 'size',            jk), 
                    (  'win', 'sum',            jk)]].fillna(0)
        data_i.columns=['place', 'run', 'win']
        # 지난 31일 동안의 입상, 출전, 승리 횟수를 누적 합으로 계산합니다.

        data_i['plc_sum']=data_i['place'].rolling(11).apply(lambda x: x[:-1].sum())
        data_i['win_sum']=data_i['win'].rolling(11).apply(lambda x: x[:-1].sum())
        data_i['run_sum']=data_i['run'].rolling(11).apply(lambda x: x[:-1].sum())
         # 지난 31일 동안의 입상, 출전, 승리 횟수를 누적 합으로 계산합니다.
        data_i['plc_rate'] = data_i['plc_sum']/data_i['run_sum']
        data_i['win_rate'] = data_i['win_sum']/data_i['run_sum']
        data_i['horse_id']=jk
        # 지난 3, 5, 10일 동안의 입상률과 승률의 변화를 계산합니다.
        for col in [ 'plc_rate', 'win_rate']:
            for i in [2,3,4,5,6,7,8,9,10]:
                data_i[f'{col}_diff{i}'] = data_i[col].diff(i)
        #dfs_pre.append(data_i.reset_index()[['trainer', 'race_date', 'win_sum', 'plc_sum', 'run_sum', 'plc_rate', 'win_rate']])
        dfs_pre.append(data_i.reset_index())
        
    # 조키별 데이터 프레임을 하나로 병합합니다.
    df_pre = pd.concat(dfs_pre)
    # 새로운 컬럼 이름을 설정합니다.

    new_cols = []
    for col in df_pre.columns:
        if col in ['horse_id', 'race_date']:
            new_cols.append(col)
        else:
            new_cols.append(f'pre11_horse_id_{col}')
        
    # print(new_cols)
    df_pre.columns = new_cols

    #--------
    # print(cnt.info)
    # 최종적으로, 위에서 생성한 데이터프레임을 원본 `start_table`에 병합하여, 말의 과거 성적 정보를 포함시킵니다.
    start_table = start_table.merge(df_pre, on=['horse_id', 'race_date'], how='left')
    # print(start_table.shape)
    start_table.drop(columns=['pre_race_date'], inplace=True)

    # 병합된 데이터를 기존의 `start_table` 데이터프레임에 병합합니다.
    start_table.replace([-np.inf, np.inf], np.nan, inplace=True)

    start_table.isna().sum()[start_table.isna().sum()>0]

    start_table.fillna(0, inplace=True)
    start_table.isna().sum()[start_table.isna().sum()>0]
    return start_table



def getData():
    random_seed=1234
    pl.seed_everything(random_seed)
    ## 정확하게 파악안됨
    start_table = pd.read_csv('../csv/total.csv')
    start_table['target'] = (start_table['position_at_finish']==1).astype(int)
    # start_table['target'].value_counts()
    start_table['race_date'] = pd.to_datetime(start_table['race_date'])
    start_table['weekday'] = start_table['race_date'].dt.weekday
    start_table['month'] = start_table['race_date'].dt.month
    start_table = getHorse(start_table)
    start_table = getJockey(start_table)
    start_table = getTrainer(start_table)

    # 최종 데이터프레임을 CSV 파일로 저장합니다.

    return start_table


def getDfData(start_table,train_start, train_end,test_start, test_end ):
    df_train = start_table[(start_table['race_date']>=train_start) & (start_table['race_date']<train_end)].copy()
    df_test = start_table[(start_table['race_date']>=test_start) & (start_table['race_date']<test_end)].copy()
    return df_train,df_test


def getKfold(df_train,train_start, train_end,feats):
    train_start_1,train_start_2,train_start_3 = divide_date_range_into_thirds(train_start, train_end)
    train_date = [train_start_1,train_start_2,train_start_3]
    print(train_date)
    kfold = []
        
    for i in range(3): 
        X_train = df_train.loc[(df_train['race_date']>=train_start) & (df_train['race_date']<train_date[i]), feats]
        y_train = df_train.loc[(df_train['race_date']>=train_start) & (df_train['race_date']<train_date[i]), ['target']]
        X_test = df_train.loc[(df_train['race_date']>=train_date[i]) & (df_train['race_date']<train_end), feats]
        y_test = df_train.loc[(df_train['race_date']>=train_date[i]) & (df_train['race_date']<train_end), ['target']]
        kfold.append([X_train, y_train, X_test, y_test])

    return kfold

from datetime import datetime, timedelta

def divide_date_range_into_thirds(start_date_str, end_date_str):
    # 날짜 형식을 datetime 객체로 변환
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # 총 기간을 계산
    total_days = (end_date - start_date).days
    
    # 기간을 3등분
    one_four= total_days // 4
    two_four = 2 * one_four
    three_four = 3 * one_four

    
    # 각 구간의 시작 날짜 계산
    first = start_date + timedelta(days=one_four)
    second = start_date + timedelta(days=two_four)
    third = start_date + timedelta(days=three_four)

    # 세 구간의 시작 날짜 반환
    return first.strftime("%Y-%m-%d"), second.strftime("%Y-%m-%d"), third.strftime("%Y-%m-%d")




def getFeats():
    return [
    "distance", "course_condition_1", "course_condition_2","course_condition_3", "course_condition_4", "course_condition_5", 
    "start_number",
    # 'win_odd','place_odd',
    "horse_age","burden_weight", "horse_weight", "horse_weight_diff", 
    "total_horse_appearances", "total_horse_first_place","total_horse_second_place", "total_horse_third_place", 
    "yearly_horse_appearances", "yearly_horse_first_place","yearly_horse_second_place", "yearly_horse_third_place", 
    "horse_grade", "trainer_age",
    "trainer_career", "total_trainer_appearances", "total_trainer_first_place", "total_trainer_second_place","total_trainer_third_place", 
    "yearly_trainer_appearances", "yearly_trainer_first_place", "yearly_trainer_second_place","yearly_trainer_third_place", 
    "jockey_age", "jockey_career", "jockey_weight", 
    "total_jockey_appearances","total_jockey_first_place", "total_jockey_second_place", "total_jockey_third_place", 
    "yearly_jockey_appearances","yearly_jockey_first_place", "yearly_jockey_second_place", "yearly_jockey_third_place", 
    "birth_korea", "birth_other","male", "female", "castration", 
    "weekday", "month", 

    "horse_id",
    "pre_horse_cnt", "pre_horse_win_cnt", "pre_horse_place_cnt", 
    "pre_horse_cnt_cumsum", "pre_horse_win_rate","pre_horse_place_rate", 
    "pre11_horse_id_run",
    "pre11_horse_id_plc_sum","pre11_horse_id_win_sum","pre11_horse_id_run_sum",
    "pre11_horse_id_plc_rate","pre11_horse_id_win_rate",
    "pre11_horse_id_plc_rate_diff2","pre11_horse_id_plc_rate_diff3","pre11_horse_id_plc_rate_diff4","pre11_horse_id_plc_rate_diff5",
    "pre11_horse_id_plc_rate_diff6","pre11_horse_id_plc_rate_diff7","pre11_horse_id_plc_rate_diff8","pre11_horse_id_plc_rate_diff9","pre11_horse_id_plc_rate_diff10",
    "pre11_horse_id_win_rate_diff2","pre11_horse_id_win_rate_diff3","pre11_horse_id_win_rate_diff4","pre11_horse_id_win_rate_diff5",
    "pre11_horse_id_win_rate_diff6","pre11_horse_id_win_rate_diff7","pre11_horse_id_win_rate_diff8","pre11_horse_id_win_rate_diff9","pre11_horse_id_win_rate_diff10",


    "trainer_id",
    "pre_trainer_cnt", "pre_trainer_win_cnt", "pre_trainer_place_cnt", 
    "pre_trainer_cnt_cumsum", "pre_trainer_win_rate", "pre_trainer_place_rate",
    "pre11_trainer_run",
    "pre11_trainer_plc_sum","pre11_trainer_win_sum","pre11_trainer_run_sum",
    "pre11_trainer_plc_rate","pre11_trainer_win_rate",
    "pre11_trainer_plc_rate_diff2","pre11_trainer_plc_rate_diff3","pre11_trainer_plc_rate_diff4","pre11_trainer_plc_rate_diff5",
    "pre11_trainer_plc_rate_diff6","pre11_trainer_plc_rate_diff7","pre11_trainer_plc_rate_diff8","pre11_trainer_plc_rate_diff9","pre11_trainer_plc_rate_diff10",
    "pre11_trainer_win_rate_diff2","pre11_trainer_win_rate_diff3","pre11_trainer_win_rate_diff4","pre11_trainer_win_rate_diff5",
    "pre11_trainer_win_rate_diff6","pre11_trainer_win_rate_diff7","pre11_trainer_win_rate_diff8","pre11_trainer_win_rate_diff9","pre11_trainer_win_rate_diff10",


    "jockey_id",
    "pre_jockey_cnt", "pre_jockey_win_cnt", "pre_jockey_place_cnt", 
    "pre_jockey_cnt_cumsum", "pre_jockey_win_rate", "pre_jockey_place_rate",
    "pre11_jockey_run",
    "pre11_jockey_plc_sum","pre11_jockey_win_sum","pre11_jockey_run_sum",
    "pre11_jockey_plc_rate","pre11_jockey_win_rate",
    "pre11_jockey_plc_rate_diff2","pre11_jockey_plc_rate_diff3","pre11_jockey_plc_rate_diff4","pre11_jockey_plc_rate_diff5",
    "pre11_jockey_plc_rate_diff6","pre11_jockey_plc_rate_diff7","pre11_jockey_plc_rate_diff8","pre11_jockey_plc_rate_diff9","pre11_jockey_plc_rate_diff10",
    "pre11_jockey_win_rate_diff2","pre11_jockey_win_rate_diff3","pre11_jockey_win_rate_diff4","pre11_jockey_win_rate_diff5",
    "pre11_jockey_win_rate_diff6","pre11_jockey_win_rate_diff7","pre11_jockey_win_rate_diff8","pre11_jockey_win_rate_diff9","pre11_jockey_win_rate_diff10",

]


def getSpace():
    return  { 
                'n_estimators': hp.choice('n_estimators', range(600, 1200, 1)),#num_boost_round
                'boosting_type':hp.choice('boosting_type', ['gbdt']),#boosting
                'objective':hp.choice('objective', ['binary'] ),
                'metric':hp.choice('metric', ['auc']),
                'learning_rate':  hp.choice('learning_rate', np.arange(0.01, 0.1, 0.01)), 
                'colsample_bytree': hp.choice('colsample_bytree', np.round(np.arange(0.1, 0.86, 0.05),3)), #feature_fraction
                'max_depth': hp.choice('max_depth', range(1, 3, 1)), #int type
                'min_child_samples': hp.choice('min_child_samples',range(10, 501, 5)), #min_data_in_leaf
                'reg_alpha':hp.choice('reg_alpha', [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,15]),#lambda_l1
                'reg_lambda':hp.choice('reg_lambda', [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,15]),#lambda_l2
                'max_bin':hp.choice('max_bin', range(20, 200, 5)),
                'min_data_in_bin':hp.choice('min_data_in_bin', range(5, 100, 5)),
                'subsample':hp.choice('subsample', np.round(np.arange(0.1, 0.96, 0.05),3)), #bagging_fraction
                'subsample_freq':hp.choice('subsample_freq', range(1, 20, 1)),#bagging_freq
                #max number of leaves in one tree. 1 < num_leaves <= 131072. classes< num_leaves< 2^max_depth  
                'num_leaves':hp.choice('num_leaves', range(31, 300, 5)),#max_leaves. 
                'random_state':hp.choice('random_state', [1234]),
                'n_jobs':hp.choice('n_jobs', [4]),#nthread 
                'scale_pos_weight':hp.choice('scale_pos_weight', [8]),# scale_pos_weight 
                #'min_split_gain':hp.choice('min_split_gain', [0.0]), #min_gain_to_split
                #'min_child_weight':hp.choice('min_child_weight', [0.001]),   #min_sum_hessian_in_leaf
                #'subsample_for_bin':hp.choice('subsample_for_bin', [200000]),   #bin_construct_sample_cnt 
                #'importance_type':hp.choice('importance_type', ['split']),   
            }


def train_trees(X_train, y_train, num_round=100, params={} ):
    params['verbosity'] = -1
    dtrain = lgb.Dataset(X_train, y_train)
    
    tree_model = lgb.train(params,
                dtrain,
                num_boost_round=num_round)
    
    
    del dtrain
    gc.collect()

    return tree_model



loss_dict = []
def score(params):
    num_boost_round = params['n_estimators']
    params_ = copy.deepcopy(params)
    del params_['n_estimators']
    losses = []
    for X_train,y_train,  X_test,y_test  in kfold:
        #-----start: train trees---------------------------------------
        tree_model = train_trees(X_train, y_train['target'].values,
                                 num_round=num_boost_round, 
                                 params=params_
                                )
        y_preds = tree_model.predict(X_test, num_iteration=tree_model.best_iteration)
        loss = roc_auc_score(y_test['target'], y_preds)
#         loss = f1_score(y_test['target'].values, (y_preds>=0.9).astype(int))
        losses.append(loss)
        #-----end: train trees---------------------------------------
        
    loss = np.mean(losses)
    #print(loss)
    loss_dict.append({'params': params, 'losses': losses, 'mean_loss': loss})
    return {'loss': -loss, 'status': STATUS_OK}


def optimize(space,k, evals, cores, trials, optimizer=tpe.suggest, random_state=1234, n_startup_jobs=10):
    global kfold
    kfold=k
    algo = partial(optimizer, n_startup_jobs=n_startup_jobs)
    best = fmin(score, space, algo=algo, max_evals=evals, trials = trials)
    bestparams = space_eval(space, best)
    return bestparams

def bar(data, caption, sub_title='', figsize=(12, 6)):
    if data.index.dtype in ['int64', 'int32', 'float64', 'float32']:
        data.index = data.index.astype(str)

    plt.figure(figsize=figsize)
    plt.style.use('seaborn-whitegrid')
    plt.barh(data.index.values, data.values, color="lightseagreen")
    for i, v in enumerate(data.values):
        plt.text(v + 3.9, i + .025, f'{v:.0f}', fontsize=12, color="teal",fontweight='bold')#deepskyblue#dodgerblue
        
        
    plt.suptitle(caption,fontsize=15, y=0.89, x=0.2)
    plt.title(sub_title,fontsize=13, y=0.96, loc='left')    
    #plt.xlabel(data.name, fontsize=13)
    plt.xlabel('', fontsize=13)
    plt.xticks([])
    plt.grid(visible=True, which='major', axis='both', color='lightgray', linestyle='-', linewidth=0.2)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.gca().spines[['bottom', 'left']].set_edgecolor('whitesmoke')
    plt.rc('ytick', labelsize=13, ) 
    
    plt.show()


def getScores(models_dict,feats):
    scores_dfs = []

    for iter_i, tree_model in models_dict.items():
        ## tree_model.feature_importance() =   함수를 사용하여 각 특성의 중요도 
        scores_ = pd.DataFrame(data = {'feat': feats, 'gain': tree_model.feature_importance()})
        # if iter_i==1:
        #     print(scores_)

        ## gain_cut = 가장 중요한 상위 5개 특성의 평균 중요도의 1%(gain_cut)를 기준으로 설정
        gain_cut = 0.01*scores_['gain'].sort_values(ascending=False).head(5).mean()
        # print(gain_cut)
        # 1.49
        # 1.06
        # 1.44

        ss = scores_[scores_['gain']>gain_cut].sort_values(by='gain')
        # if iter_i==1:
        #     ## gain_cut으로 걸러내고 남은 22개 
        #     print(ss.head(100))
        ss['gain_z'] = (ss['gain'] - ss['gain'].min())/(ss['gain'].quantile(q=0.95)-ss['gain'].quantile(q=0.05))
        # if iter_i==1:
        #     ## 22개중 가장 작은값을 0, 으로 기준한 백분율
        #     print(ss.head(22))
        ss.reset_index(drop=True, inplace=True)
        ss['rank']=range(ss.shape[0], 0, -1)
        scores_dfs.append(ss)
        # if iter_i==1:
            ## 22개 기준 우선순위 rank 적용
            # print(ss.head(22))

        all_scores = pd.concat(scores_dfs, axis=0)
        # print(all_scores)

    return all_scores


def getImp(all_scores):
    imp = pd.pivot_table(all_scores, 
                index='feat', 
                values=['gain_z'], 
                aggfunc={'gain_z':[np.mean, np.size],
                        }
                )
    imp.columns = ['score', 'cnt']
    imp.sort_values(by='score', ascending=True, inplace=True)

    imp['score'] = 100*imp['score']
    ## all_scores 로 구한 22개 변수 3쌍중 평균 및 중복 카운트 계산하여 총 중복되지 않는 40여개 뽑아냄
    # print(imp)
    return imp



def getRangeNum():
    return 3

class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=5.0):

        self.thresh = thresh

   

    def fit(self, X):
        print('ReduceVIF fit')

        return self

    def transform(self, X):
        print('ReduceVIF transform')
        columns = X.columns.tolist()

        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=5.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            vif = np.array(vif)
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = list(vif).index(vif[vif>thresh][0])
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X


