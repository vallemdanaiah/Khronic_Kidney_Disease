from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import seaborn as sns


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

#userloginform
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print('Login ID = ', loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})



#rewdataset   
def DatasetView(request):
    path = settings.MEDIA_ROOT + '//' + 'kidney_disease.csv'
    df = pd.read_csv(path, nrows=100)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})


#preprocessdataset
def preprocess(request):
    path = settings.MEDIA_ROOT + '//' + 'danaiahpreposedataset-1.csv'
    df = pd.read_csv(path, nrows=300)
    df = df.to_html
    return render(request, 'users/viewdataset-1.html', {'data': df})


#formance matrix in different algrothems
def ML(request):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    #from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense    
    from keras.wrappers.scikit_learn import KerasRegressor
    #from sklearn.model_selection import train_test_split
    from django.conf import settings
    path = settings.MEDIA_ROOT + '//' + 'danaiahpreposedataset-1.csv'    
    df = pd.read_csv(path)
    df['class'] = df['class'].map({'ckd': 1, 'notckd': 0})
    X = df.drop('class',axis=1)
    Y= df['class']
    print(Y)
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=101)#shuffle=True
    Y_train.value_counts()
    Y_test.value_counts()
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train,Y_train)
    y_pred =model.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(Y_test, y_pred) * 100
    print('Accuracy:', accuracy)
    from sklearn.metrics import precision_score
    precision = precision_score(Y_test, y_pred) * 100
    print('Precision Score:', precision)
    from sklearn.metrics import recall_score
    recall = recall_score(Y_test, y_pred) * 100
    print('recall_score:',recall)
    from sklearn.metrics import f1_score
    f1score = f1_score(Y_test, y_pred) * 100
    print('f1score:',f1score)



    #algrothem2
    from sklearn.tree import DecisionTreeClassifier
    model1 = DecisionTreeClassifier()
    model1.fit(X_train, Y_train)
    y_pred2 = model1.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy1 = accuracy_score(Y_test, y_pred2) * 100
    print('Accuracy1:', accuracy1)
    from sklearn.metrics import precision_score
    precision1 = precision_score(Y_test, y_pred2) * 100
    print('precision1:',precision1)
    from sklearn.metrics import recall_score
    recall1 = recall_score(Y_test, y_pred2) * 100
    print('recall1:',recall1)  
    from sklearn.metrics import f1_score
    f1score1 = f1_score(Y_test, y_pred) * 100
    print('f1score1:',f1score1) 

    

    #algrothem-3
    from sklearn.ensemble import RandomForestClassifier
    model2= RandomForestClassifier(n_estimators=10,criterion='entropy',)    
    model2.fit(X_train, Y_train)
    y_pred2 = model2.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy2 = accuracy_score(Y_test, y_pred2) * 100
    print('Accuracy2:', accuracy2)
    from sklearn.metrics import precision_score
    precision2 = precision_score(Y_test, y_pred2) * 100
    print('precision2:',precision2)
    from sklearn.metrics import recall_score
    recall2 = recall_score(Y_test, y_pred2) * 100
    print('recall2:',recall2)  
    from sklearn.metrics import f1_score
    f1score2 = f1_score(Y_test, y_pred) * 100
    print('f1score2:',f1score2) 



    #algrothem-4    
    from sklearn.svm import SVC
    model3= SVC()    #n_estimators=10,criterion='entropy'
    model3.fit(X_train, Y_train)
    y_pred3 = model3.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy3 = accuracy_score(Y_test, y_pred3) * 100
    print('Accuracy3:', accuracy3)
    from sklearn.metrics import precision_score
    precision3 = precision_score(Y_test, y_pred3) * 100
    print('precision3:',precision3)
    from sklearn.metrics import recall_score
    recall3 = recall_score(Y_test, y_pred3) * 100
    print('recall3:',recall3)
    from sklearn.metrics import f1_score
    f1score3 = f1_score(Y_test, y_pred3) * 100
    print('f1score3:',f1score3) 



    #algrothem-5 
    from sklearn.ensemble import GradientBoostingClassifier
    model4= GradientBoostingClassifier()    #n_estimators=10,criterion='entropy'
    model4.fit(X_train, Y_train)
    y_pred4 = model4.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy4 = accuracy_score(Y_test, y_pred4) * 100
    print('Accuracy4:', accuracy4)
    from sklearn.metrics import precision_score
    precision4 = precision_score(Y_test, y_pred4) * 100
    print('precision4:',precision4)
    from sklearn.metrics import recall_score
    recall4 = recall_score(Y_test, y_pred4) * 100
    print('recall4:',recall4) 
    from sklearn.metrics import f1_score
    f1score4 = f1_score(Y_test, y_pred4) * 100
    print('f1score4:',f1score4) 




    #algrothem-6
    from xgboost import XGBClassifier
    model5= XGBClassifier()    #n_estimators=10,criterion='entropy'
    model5.fit(X_train, Y_train)
    y_pred5 = model5.predict(X_test)
    from sklearn.metrics import accuracy_score
    accuracy5 = accuracy_score(Y_test, y_pred5) * 100
    print('Accuracy5:', accuracy5)
    from sklearn.metrics import precision_score
    precision5 = precision_score(Y_test, y_pred5) * 100
    print('precision5:',precision5)
    from sklearn.metrics import recall_score
    recall5 = recall_score(Y_test, y_pred5) * 100
    print('recall5:',recall5) 
    from sklearn.metrics import f1_score
    f1score5 = f1_score(Y_test, y_pred5) * 100
    print('f1score5:',f1score5) 

#performancematrix in different algrothmes
    accuracy = {'LR': accuracy, 'DTC': accuracy1, 'RFC': accuracy2,'SVC':accuracy3,'GBC':accuracy4,'XGBC':accuracy5}
    precision = {'LR': precision, 'DTC': precision1, 'RFC': precision2,'SVC':precision3,'GBC':precision4,'XGBC':precision5}
    recall = {'LR': recall, 'DTC': recall1, 'RFC': recall2,'SVC':recall3,'GBC':recall4,'XGBC':recall5}
    f1score = {'LR':f1score,'DTC':f1score1,'RFC':f1score2,'SVC':f1score3,'GBC':f1score4,'XGBC':f1score5}
    return render(request, 'users/ML.html',
                  {"accuracy": accuracy, "precision": precision, "recall":recall, 'f1score':f1score})

#predication from funcation
def predictTrustWorthy(request):
    if request.method == 'POST':
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from django.conf import settings
        age=request.POST.get("age")  
        bp=request.POST.get("bp")
        sg=request.POST.get("sg")
        al=request.POST.get("al")
        su=request.POST.get("su")
        sod=request.POST.get("sod")
        pot=request.POST.get("pot")
        hemo=request.POST.get("hemo")
        wbcc=request.POST.get("wbcc")
        rbcc=request.POST.get("rbcc")
        skinthickness=request.POST.get("skinthickness")
        insulin=request.POST.get("insulin")
        BMI=request.POST.get("BMI")        
        path = settings.MEDIA_ROOT + '//' + 'danaiahpreposedataset-1.csv'
        df = pd.read_csv(path)
        data = df
        data = data.dropna()
        data['class'] = data['class'].replace(['ckd','notckd'],[0,1])
        #data = data.drop(['rbc','pc','pcc','ba','bgr','bu','sc','pcv','htn','dm','cad','appet','pe','ane'],axis = 1)
        from sklearn.ensemble import RandomForestClassifier
        OBJ = RandomForestClassifier(n_estimators=10,criterion='entropy',)
        X = data.iloc[:, :-1]
        y = data['class']
        X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size= 0.2, random_state=101)#shuffle=True
        test_set = [age,bp,sg,al,su,sod,pot,hemo,wbcc,rbcc,skinthickness,insulin,BMI]
        OBJ.fit(X_train.values,Y_train)
        print(test_set)
        y_pred = OBJ.predict([test_set])
        print(y_pred)        
        if y_pred == 1:
            msg =  'chronic disease'
        else:
            msg =  ' no diesease'
        return render(request,"users/prediction.html",{"msg":msg})
    else:
        return render(request,'users/prediction.html',{})