# Adi Paz
# Neriya Mazzuz
# Gil Shamay 
seed = 415

import time
import pandas as pd
import zipfile
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
from scipy import stats
import lightgbm as lgb
import numpy as np
from sklearn import preprocessing

#  LightGBM vs  RandomForest for CTR
####################################################################################
# the DATA
# Data columns (total 23 columns):
#  #   Column                   Non-Null Count   Dtype
# ---  ------                   --------------   -----
#  0   page_view_start_time     462734 non-null  int64
#  1   user_id_hash             462734 non-null  object
#  2   target_id_hash           462734 non-null  object
#  3   syndicator_id_hash       462734 non-null  object //the customer
#  4   campaign_id_hash         462734 non-null  object
#  5   empiric_calibrated_recs  462734 non-null  float64 //num of clicks target got - calibrated low/high - float //
#  6   empiric_clicks           462734 non-null  float64 //num of clicks target got - actual  - int
#  7   target_item_taxonomy     462734 non-null  object //BUSINESS/SPORT/...
#  8   placement_id_hash        462734 non-null  object //affect the calibration
#  9   user_recs                462734 non-null  float64 // user actual saw
#  10  user_clicks              462734 non-null  float64 // user actual clicked
#  11  user_target_recs         462734 non-null  float64 //how many he saw this
#  12  publisher_id_hash        462734 non-null  object //the website
#  13  source_id_hash           462734 non-null  object //web actual page
#  14  source_item_type         462734 non-null  object //type of page
#  15  browser_platform         462734 non-null  object //OS
#  16  os_family                462734 non-null  int64
#  17  country_code             462727 non-null  object
#  18  region                   462724 non-null  object
#  19  day_of_week              462734 non-null  int64
#  20  time_of_day              462734 non-null  int64
#  21  gmt_offset               462734 non-null  int64
#  22  is_click                 462734 non-null  float64
# dtypes: float64(6), int64(5), object(12)
###############################################################################################


submit = False # test on test dat as uploaded to https://www.kaggle.com/c/bgu-rs/data See test_file_3/test_file.csv
maxFiles = 10
pd.options.display.width = 0


def readCSVFromZip(archive, file):
    readBeginTime = time.time()
    fileData = archive.read(file.filename)
    print("read Zip file took [" + str(time.time() - readBeginTime) + "]")
    s = str(fileData, 'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)
    return df


def readTrainDataRF():
    beginTime = time.time()
    archive = zipfile.ZipFile('./data/bgu-rs.zip', 'r')
    i=0
    for file in archive.filelist:
        if "part-" in file.filename and ".csv" in file.filename and i < maxFiles:
            fileBeginTime = time.time()
            if i==0:
                df = readCSVFromZip(archive, file)
            else:
                new_df = readCSVFromZip(archive, file)
                df = pd.concat([df, new_df])

            print("file: " + str(i) + " handle time[" + str(time.time() - fileBeginTime)+ "]")
            i = i+1

    trainDf, testDf = train_test_split(df, train_size=0.99, random_state=seed)

    trainDf = trainDf.dropna()  # todo: do we need this?
    testDf = testDf.dropna()
    trainY = trainDf.pop('is_click')
    trainX = trainDf
    trainX, ce_target_encoder = preprocessDataRF(trainX, trainY)
    print(trainX.describe())

    testY = testDf.pop('is_click')
    testX = testDf
    testX, ce_target_encoder = preprocessDataRF(testX, ce_target_encoder=ce_target_encoder)
    print(testX.describe())
    print("Number of rows in train: " + str(trainX.shape[0]))
    print("Time taken: [" + str(time.time() - beginTime) + "]")
    return trainX, testX, trainY, testY, ce_target_encoder

    
def readTrainData():
    beginTime = time.time()
    archive = zipfile.ZipFile('./data/bgu-rs.zip', 'r')
    i=0
    for file in archive.filelist:
        if "part-" in file.filename and ".csv" in file.filename and i < maxFiles:
            fileBeginTime = time.time()
            if i==0:
                df = readCSVFromZip(archive, file)
            else:
                new_df = readCSVFromZip(archive, file)
                df = pd.concat([df, new_df])

            print("file: " + str(i) + " handle time[" + str(time.time() - fileBeginTime)+ "]")
            i = i+1

    trainDf, testDf = train_test_split(df, train_size=0.99, random_state=seed)

    trainDf = trainDf.dropna()  # todo: do we need this?
    testDf = testDf.dropna()
    trainY = trainDf.pop('is_click')
    trainX = trainDf
    trainX, ce_target_encoder = preprocessData(trainX, trainY)
    print(trainX.describe())

    testY = testDf.pop('is_click')
    testX = testDf
    testX, ce_target_encoder = preprocessData(testX, ce_target_encoder=ce_target_encoder)
    print(testX.describe())
    print("Number of rows in train: " + str(trainX.shape[0]))
    print("Time taken: [" + str(time.time() - beginTime) + "]")
    return trainX, testX, trainY, testY, ce_target_encoder


def target_encode(X, y, categorical_columns):
    ce_target_encoder = ce.TargetEncoder(cols=categorical_columns, min_samples_leaf=10)
    ce_target_encoder.fit(X, y)
    X = ce_target_encoder.transform(X)
    return X, ce_target_encoder


def preprocessDataRF(df, y=None, ce_target_encoder=None):
    df , test= preprocessData(df, y, ce_target_encoder)
    df = df.apply(preprocessing.LabelEncoder().fit_transform)
    return df, None


def preprocessData(df, y=None, ce_target_encoder=None):
    fitBeginTime = time.time()
    df['os_is_six'] = [1 if x == 6 else 0 for x in df['os_family']]
    categorical_columns = ['user_id_hash', 'target_id_hash', 'syndicator_id_hash', 'campaign_id_hash',
                           'placement_id_hash', 'publisher_id_hash', 'source_id_hash',
                           'source_item_type', 'browser_platform', 'country_code', 'region',
                           'os_family', 'day_of_week']
    for column_name in categorical_columns:
        df[column_name] = pd.Categorical(df[column_name])

    """""
    if ce_target_encoder==None:
        df, ce_target_encoder = target_encode(df, y, categorical_columns)
    else:
        df = ce_target_encoder.transform(df)
    """

    print("transformDataToX_Y took[" + str(time.time() - fitBeginTime) + "]")

    df.pop('page_view_start_time')

    df['night'] = [1 if x<=6 else 0 for x in df['time_of_day']]
    df['evening'] = [1 if x >= 17 else 0 for x in df['time_of_day']]

    # time + gmt_offset --> new time ; remove GMT // (save time.. ?)
    df['time_of_day'] = df['time_of_day'] + (df['gmt_offset'] / 100.0)
    df.pop('gmt_offset')

    # user click rate, with an option that the user is a cold start
    df['user_click_rate'] = (df['user_clicks'] + 1) / (df['user_recs'] + 1)
    df['user_click_rate_pow'] = np.power(df['user_click_rate'], 2)

    # df['target_item_taxonomy_list'] = df.apply(lambda row : row['target_item_taxonomy'].split('~'), axis = 1)
    # categories = df['target_item_taxonomy_list'].tolist()
    # categories = sum(categories, [])
    # freqs = {i: categories.count(i) for i in set(categories)}
    # sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    taxonomy_categories = ['LIFE', 'BUSINESS', 'ENTERTAINMENT', 'HEALTH', 'TECH', 'AUTOS', 'FOOD', 'SPORTS', 'MUSIC',
                           'PETS', 'NEWS', 'FASHION', 'FOOTBALL']
    for column_name in taxonomy_categories:
        df[column_name] = [1 if column_name in x else 0 for x in df['target_item_taxonomy']]

    # df = pd.concat([df, df['target_item_taxonomy'].str.get_dummies('~').add_prefix('C_')], axis=1)
    # TODO: We should  find how to use this data
    df.pop('target_item_taxonomy')
    #df.pop('user_id_hash')
    #df.pop('target_id_hash')
    #df.pop('syndicator_id_hash')
    #df.pop('campaign_id_hash')
    #df.pop('placement_id_hash')
    #df.pop('publisher_id_hash')
    #df.pop('source_id_hash')
    # return df, ce_target_encoder

    #this is needed for random forests ! RandomForestClassifier - it kills the auc for the lgbm
    #df = df.apply(preprocessing.LabelEncoder().fit_transform)
    return df, None




def builedModelRF(trainX, trainY):
    RF = RandomForestClassifier()
    RF.fit(trainX,trainY)
    return RF


def builedModelLightGBM(trainX, trainY):
    categorical_features = ['user_id_hash', 'target_id_hash', 'syndicator_id_hash', 'campaign_id_hash',
                           'placement_id_hash', 'publisher_id_hash', 'source_id_hash',
                           'source_item_type', 'browser_platform', 'country_code', 'region',
                           'os_family', 'day_of_week']
    taxonomy_categories = ['LIFE', 'BUSINESS', 'ENTERTAINMENT', 'HEALTH', 'TECH', 'AUTOS', 'FOOD', 'SPORTS', 'MUSIC',
                           'PETS', 'NEWS', 'FASHION', 'FOOTBALL']
    categorical_features = categorical_features + taxonomy_categories
    categorical_features = [c for c, col in enumerate(trainX.columns) if col in categorical_features]
    train_data = lgb.Dataset(trainX, label=trainY, categorical_feature=categorical_features)

    #
    # Train the model
    #

    parameters = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'gbdt',
        'num_leaves': 31,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 20,
        'learning_rate': 0.05,
        'verbose': 2
    }
    fitBeginTime = time.time()
    model = lgb.train(parameters, train_data, num_boost_round=200)
    print("Training model took[" + str(time.time() - fitBeginTime) + "]")
    return model


def trainModel(X, y, model):
    fitBeginTime = time.time()
    print("start fit dataChunk")
    model.fit(X, y)
    print("fit dataChunk took[" + str(time.time() - fitBeginTime) + "]")
    return model


def normalizeResults(x):
    if (x < 0.3):
        return 0
    if (x > 0.7):
        return 1
    else:
        return x


def evaluateModel(X, y, model):
    testRes = model.predict(X)
    AUC = roc_auc_score(y, testRes)

    normRes = np.vectorize(normalizeResults)(testRes)
    AUCNorm = roc_auc_score(y, normRes)

    # todo: Check that the res data is not <0 or >1 and fix if it does
    print('test: AUC[' + str(AUC) + ']' + 'test: AUCNorm[' + str(AUCNorm) + ']' + str(stats.describe(testRes)))
    return AUC

def evaluateModelRF(X, y, model):
    testResProb = model.predict_proba(X)
    testRes = list(map(lambda x : x[1], testResProb))
    AUC = roc_auc_score(y, testRes)

    normRes = np.vectorize(normalizeResults)(testRes)
    AUCNorm = roc_auc_score(y, normRes)

    # todo: Check that the res data is not <0 or >1 and fix if it does
    print('test: AUC[' + str(AUC) + ']' + 'test: AUCNorm[' + str(AUCNorm) + ']' + str(stats.describe(testRes)))
    return AUC


def loadUncompressed(path):
    chunksNum = 0
    beginTime = time.time()
    data = None
    pd.read_csv(path, chunksize=20000)
    for dataChunk in pd.read_csv(path, chunksize=20000):
        if (data is None):
            data = dataChunk
        else:
            data = data.append(dataChunk, ignore_index=True)
        chunksNum += 1
    took = time.time() - beginTime
    print("LOAD: chunksNum[" + str(chunksNum) + "]took[" + str(took) + "]data[" + str(len(data)) + "]")
    return data


def run():
    # read the data and fit
    print("-------------------------------")
    trainX, testX, trainY, testY, ce_target_encoder = readTrainData()
    trainXRF, testXRF, trainYRF, testYRF, ce_target_encoder = readTrainDataRF() # todo: not good - data is read twice

    # trainXRF = trainX.copy()
    # testXRF = testX.copy()
    # trainYRF = trainY.copy()
    # # testYRF = testY.copy()
    # le = preprocessing.LabelEncoder()
    # trainXRF = trainXRF.apply(le.fit_transform)
    # testXRF = testXRF.apply(le.fit_transform)

    modelLRF = None
    modelLGBM = None

    modelLRF = builedModelRF(trainXRF, trainYRF)
    modelLGBM = builedModelLightGBM(trainX, trainY)

    if modelLRF is not None:
        aucRF = evaluateModelRF(testXRF, testYRF, modelLRF)
        print('AUC RF[' + str(aucRF) + ']')

    if modelLGBM is not None:
        aucLGBM = evaluateModel(testX, testY, modelLGBM)
        print('AUC LGBM[' + str(aucLGBM) + ']')

run()
print(" ---- Done ---- ")

exit(0)
