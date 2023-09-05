
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from django.conf import settings



path = settings.MEDIA_ROOT + '//' + 'kidney_disease.csv'
def logisic(request):
    df = pd.read_csv(path)
    data = df
    data.head()
    data['class'] = data['class'].map({'ckd': 1, 'notckd': 0})
    data['htn'] = data['htn'].map({'yes': 1, 'no': 0})
    data['dm'] = data['dm'].map({'yes': 1, 'no': 0})
    data['cad'] = data['cad'].map({'yes': 1, 'no': 0})
    data['appet'] = data['appet'].map({'good': 1, 'poor': 0})
    data['ane'] = data['ane'].map({'yes': 1, 'no': 0})
    data['pe'] = data['pe'].map({'yes': 1, 'no': 0})
    data['ba'] = data['ba'].map({'present': 1, 'notpresent': 0})
    data['pcc'] = data['pcc'].map({'present': 1, 'notpresent': 0})
    data['pc'] = data['pc'].map({'abnormal': 1, 'normal': 0})
    data['rbc'] = data['rbc'].map({'abnormal': 1, 'normal': 0})

    data['class'].value_counts()
    #plt.figure(figsize=(19, 19))
    # sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    print(data.shape)
    print(data.isnull().sum())
    print(data.shape[0], data.dropna().shape[0])
    print(data.dropna(inplace=True))
    print(data.shape)
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression()
    X = data.iloc[:, :-1]
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True)
    logreg.fit(X_train, y_train)
    test_pred = logreg.predict(X_test)
    train_pred = logreg.predict(X_train)
    from sklearn import metrics
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics import confusion_matrix,mean_absolute_error,mean_squared_error,f1_score,precision_score,recall_score
    train_accuracy = accuracy_score(y_train, train_pred)
    print('Train Accuracy: ', train_accuracy)
    print('Test Accuracy: ', accuracy_score(y_test, test_pred))
    matrix= confusion_matrix(y_test, test_pred)    
    sns.heatmap(matrix,annot = True, fmt = "d")
    plt.show()
    
    mae = mean_absolute_error(y_train, train_pred)
    mse = mean_squared_error(y_train, train_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_train, train_pred))
    f1_score = f1_score(y_test,test_pred)
    precision = precision_score(y_test,test_pred)
    recall = recall_score(y_test,test_pred)
    return render(request,'rf.html',{"pred":test_pred,'train_accuracy':train_accuracy,'mae':mae,'mse':mse,'rmse':rmse,'f1_score':f1_score,'precision':precision,'recall':recall})

def decision(request):
    df = pd.read_csv('./chronic_kidney_disease_full.csv')
    data = df
    data.head()
    data['class'] = data['class'].map({'ckd': 1, 'notckd': 0})
    data['htn'] = data['htn'].map({'yes': 1, 'no': 0})
    data['dm'] = data['dm'].map({'yes': 1, 'no': 0})
    data['cad'] = data['cad'].map({'yes': 1, 'no': 0})
    data['appet'] = data['appet'].map({'good': 1, 'poor': 0})
    data['ane'] = data['ane'].map({'yes': 1, 'no': 0})
    data['pe'] = data['pe'].map({'yes': 1, 'no': 0})
    data['ba'] = data['ba'].map({'present': 1, 'notpresent': 0})
    data['pcc'] = data['pcc'].map({'present': 1, 'notpresent': 0})
    data['pc'] = data['pc'].map({'abnormal': 1, 'normal': 0})
    data['rbc'] = data['rbc'].map({'abnormal': 1, 'normal': 0})

    data['class'].value_counts()
    # plt.figure(figsize=(19, 19))
    # sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    print(data.shape)
    print(data.isnull().sum())
    print(data.shape[0], data.dropna().shape[0])
    print(data.dropna(inplace=True))
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    X = data.iloc[:, :-1]
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred=classifier.predict(X_test)
    print("prediction",y_pred)
    from sklearn.metrics import confusion_matrix
    acc=confusion_matrix(y_test,y_pred)
    print(acc)
    #return render(request,'decision.html',{"acc":acc})
    from sklearn import metrics
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics import confusion_matrix,mean_absolute_error,mean_squared_error,f1_score,precision_score,recall_score
    train_accuracy = accuracy_score(y_test, y_pred)
    print('Train Accuracy: ', train_accuracy)
    print('Test Accuracy: ', accuracy_score(y_test, y_pred))
    matrix= confusion_matrix(y_test, y_pred)    
    sns.heatmap(matrix,annot = True, fmt = "d")
    plt.show()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    f1_score = f1_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    return render(request,'decision.html',{"pred":y_pred,'train_accuracy':train_accuracy,'mae':mae,'mse':mse,'rmse':rmse,'f1_score':f1_score,'precision':precision,'recall':recall})



def randomforest(request):
    df = pd.read_csv('./chronic_kidney_disease_full.csv')
    data = df
    data.head()
    data['class'] = data['class'].map({'ckd': 1, 'notckd': 0})
    data['htn'] = data['htn'].map({'yes': 1, 'no': 0})
    data['dm'] = data['dm'].map({'yes': 1, 'no': 0})
    data['cad'] = data['cad'].map({'yes': 1, 'no': 0})
    data['appet'] = data['appet'].map({'good': 1, 'poor': 0})
    data['ane'] = data['ane'].map({'yes': 1, 'no': 0})
    data['pe'] = data['pe'].map({'yes': 1, 'no': 0})
    data['ba'] = data['ba'].map({'present': 1, 'notpresent': 0})
    data['pcc'] = data['pcc'].map({'present': 1, 'notpresent': 0})
    data['pc'] = data['pc'].map({'abnormal': 1, 'normal': 0})
    data['rbc'] = data['rbc'].map({'abnormal': 1, 'normal': 0})

    data['class'].value_counts()
    # plt.figure(figsize=(19, 19))
    # sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    print(data.shape)
    print(data.isnull().sum())
    print(data.shape[0], data.dropna().shape[0])
    print(data.dropna(inplace=True))
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',)
    X = data.iloc[:, :-1]
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("prediction", y_pred)
    from sklearn.metrics import confusion_matrix
    acc = confusion_matrix(y_test, y_pred)
    print(acc)
    #return render(request, 'randomforest.html', {"acc": acc})
    from sklearn import metrics
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.metrics import confusion_matrix,mean_absolute_error,mean_squared_error,f1_score,precision_score,recall_score
    train_accuracy = accuracy_score(y_test, y_pred)
    print('Train Accuracy: ', train_accuracy)
    print('Test Accuracy: ', accuracy_score(y_test, y_pred))
    matrix= confusion_matrix(y_test, y_pred)    
    sns.heatmap(matrix,annot = True, fmt = "d")
    plt.show()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    f1_score = f1_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    return render(request,'randomforest.html',{"pred":y_pred,'train_accuracy':train_accuracy,'mae':mae,'mse':mse,'rmse':rmse,'f1_score':f1_score,'precision':precision,'recall':recall})




