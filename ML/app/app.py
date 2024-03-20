import pandas as pd 
import copy
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score
import prepare
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, anneal, rand
import lightgbm as lgb

def getBestParam(space,kfold):
    print('------------------------------------------------------------')
    iter_dict = {}
    for iter_i in range(prepare.getRangeNum()):
        print(datetime.now(), iter_i)
        cores = 8
        n_trials = 100
        n_rand_trials = 50
        verbose = False
        trials = Trials()
        best_param = prepare.optimize(space,
                                      kfold,
                            evals = n_trials,
                            optimizer=tpe.suggest,
                            cores = cores,
                            trials = trials, random_state=1234, 
                            n_startup_jobs=n_rand_trials)
        
        iter_dict[iter_i] = best_param

    file_path = './txt/iter_dict.txt'  # 쓰고자 하는 파일의 경로
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(iter_dict))
    # df_params = pd.DataFrame(iter_dict).T
    print('------------------------------------------------------------')
    return iter_dict

def setModel(iter_dict,df_train,df_test,feats,name):
    models_dict = {}
    for iter_i, params in iter_dict.items():
        params = copy.deepcopy(params)
        num_boost_round = params['n_estimators']
        del params['n_estimators']
        tree_model = prepare.train_trees(df_train[feats], df_train['target'].values,
                                num_round=num_boost_round, 
                                params=params
                                )
        tree_model.save_model(f'./model/model_{name}_{iter_i}.txt')  # 텍스트 파일로 저장
        y_preds = tree_model.predict(df_test[feats], num_iteration=tree_model.best_iteration)
        # print(y_preds,df_test['target'])
        loss = roc_auc_score(df_test['target'], y_preds)
        models_dict[iter_i] = tree_model
        print(f'{iter_i}: auc on testing data: {loss: .04f}')
    return models_dict





def getModel(df_test,feats):
    models_dict={}
    for i in range(prepare.getRangeNum()):
        tree_model = lgb.Booster(model_file=f'./model/model_normal_{i}.txt')  # 텍스트 파일에서 불러오기
        models_dict[i] = tree_model
        y_preds = tree_model.predict(df_test[feats], num_iteration=tree_model.best_iteration)
        loss = roc_auc_score(df_test['target'], y_preds)
    return models_dict


def getBestFeats(models_dict,df_train,feats):
    all_scores= prepare.getScores(models_dict,feats)
    imp = prepare.getImp(all_scores)
    transformer = prepare.ReduceVIF(thresh=20)
    X = transformer.fit_transform(df_train[imp[imp['score']>1].sort_values(by='score', ascending=True).index.values])
    final_feats = X.columns.tolist()
    file_path = './txt/final_feats.txt'  # 쓰고자 하는 파일의 경로
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(','.join(final_feats))
    return imp,final_feats


def getBar(imp,final_feats):
    prepare.bar(imp.loc[final_feats,'score'], caption='Feature importance of selected features after VIF', 
    sub_title="", 
    figsize=(12, 14))


def getBestParams():
    file_path = './txt/iter_dict.txt'  # 읽고자 하는 파일의 경로
    iter_dict=""
    with open(file_path, 'r', encoding='utf-8') as txt:
        iter_dict=json.load(txt)
    return iter_dict


def firstStep(space,kfold,df_train,df_test,feats):
    iter_dict = getBestParam(space,kfold)
    models_dict = setModel(iter_dict,df_train,df_test,feats,'normal')
    return models_dict


def secondStep(df_train,df_test,feats):
    models_dict = getModel(df_test,feats)
    imp,final_feats = getBestFeats(models_dict,df_train,feats)
    getBar(imp,final_feats)
    iter_dict = getBestParams()
    models_dict = setModel(iter_dict,df_train,df_test,final_feats,'best')
    return models_dict
    
def loadBestFeats():
    file_path = './txt/final_feats.txt'  # 읽고자 하는 파일의 경로
    with open(file_path, 'r', encoding='utf-8') as file:
        array = [line.strip().split(',') for line in file]
    return array[0]

def getResult(final_feats,grouped):
    dataframes = []
    for name, group in grouped:
        df_pred = pd.DataFrame(data={'target': group['target'].values})
        df_pred['id'] = group['id'].values
        df_pred['round_detail'] = name[0]

        for i in range(prepare.getRangeNum()):
            tree_model = lgb.Booster(model_file=f'./model/model_best_{i}.txt')  # 텍스트 파일에서 불러오기
            # tree_model = lgb.Booster(model_file=f'./model/model_normal_{i}.txt')  # 텍스트 파일에서 불러오기
            y_preds = tree_model.predict(group[final_feats], num_iteration=tree_model.best_iteration)
            df_pred[f'pred{i}'] = y_preds

        df_pred['mean_pred'] = df_pred[[f'pred{i}' for i in range(prepare.getRangeNum())]].mean(axis=1)
        max_idx = df_pred['mean_pred'].idxmax()
        # 'B' 컬럼 업데이트
        df_pred['mean_pred_high_target']= 0
        df_pred.loc[max_idx, 'mean_pred_high_target'] = 1
        dataframes.append(df_pred)
    return dataframes

def lastStep(df_test):
    grouped = df_test.groupby(['round_detail','race_date'])
    final_feats = loadBestFeats()
    # final_feats=prepare.getFeats()
    dataframes = getResult(final_feats,grouped)
    result_df = pd.concat(dataframes, ignore_index=True)
    result_df.to_csv('../csv/result.csv',index=False)
    final = df_test.merge(result_df, on=['id', 'round_detail'], how='left')
    # final.to_csv('../csv/final.csv',index=False)

    count = ((result_df['target'] == 1) & (result_df['mean_pred_high_target'] == 1)).sum()

    print(f' result : {count}/ {len(grouped)}')
    

def main():

    start_table = prepare.getData()
    start_table.to_csv('../csv/start_table.csv',index=False)
    
main()




def play():
    train_start, train_end = '2019-01-06', '2021-12-31'
    test_start, test_end = '2022-01-01', '2022-12-31'
    _start_table = pd.read_csv('../csv/start_table.csv')
    start_table =prepare.minMaxScaler(_start_table)
    feats = prepare.getFeats()
    df_train,df_test = prepare.getDfData(start_table,train_start, train_end,test_start, test_end)
    kfold = prepare.getKfold(df_train,train_start, train_end,feats)
    space = prepare.getSpace()
    models_dict = firstStep(space,kfold,df_train,df_test,feats)
    models_dict = secondStep(df_train,df_test,feats)
    lastStep(df_test)

play()
