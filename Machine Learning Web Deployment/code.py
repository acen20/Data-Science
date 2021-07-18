# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:06.825550Z","iopub.execute_input":"2021-07-18T11:45:06.825876Z","iopub.status.idle":"2021-07-18T11:45:06.832784Z","shell.execute_reply.started":"2021-07-18T11:45:06.825848Z","shell.execute_reply":"2021-07-18T11:45:06.831734Z"}}

import pandas
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import random
import pickle

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:06.836758Z","iopub.execute_input":"2021-07-18T11:45:06.837130Z","iopub.status.idle":"2021-07-18T11:45:06.913213Z","shell.execute_reply.started":"2021-07-18T11:45:06.837103Z","shell.execute_reply":"2021-07-18T11:45:06.912438Z"}}
data = pandas.read_csv('Training.csv')
testdata = pandas.read_csv('Testing.csv')

le = LabelEncoder()

data['e_prognosis'] = le.fit_transform(data.prognosis)
testdata['e_prognosis'] = le.fit_transform(testdata.prognosis)

trainy = np.array(data['e_prognosis'])

trainx = np.array(data.drop(['prognosis','Unnamed: 133', 'e_prognosis'],axis=1))

testy = np.array(testdata['e_prognosis'])

testx = np.array(testdata.drop(['prognosis', 'e_prognosis'],axis=1))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:06.914353Z","iopub.execute_input":"2021-07-18T11:45:06.914722Z","iopub.status.idle":"2021-07-18T11:45:06.922414Z","shell.execute_reply.started":"2021-07-18T11:45:06.914689Z","shell.execute_reply":"2021-07-18T11:45:06.921713Z"}}
data[data.e_prognosis == 0].prognosis.unique()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:06.923961Z","iopub.execute_input":"2021-07-18T11:45:06.924407Z","iopub.status.idle":"2021-07-18T11:45:06.968155Z","shell.execute_reply.started":"2021-07-18T11:45:06.924374Z","shell.execute_reply":"2021-07-18T11:45:06.967141Z"}}
pca = PCA(n_components = 9, svd_solver = "full")
newtrainx = pca.fit_transform(trainx, y = trainy)
newtestx = pca.transform(testx)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:06.969703Z","iopub.execute_input":"2021-07-18T11:45:06.969968Z","iopub.status.idle":"2021-07-18T11:45:06.975426Z","shell.execute_reply.started":"2021-07-18T11:45:06.969942Z","shell.execute_reply":"2021-07-18T11:45:06.974792Z"}}
xtrain, xtest, ytrain, ytest = train_test_split(newtrainx, trainy, test_size=0.2, random_state=43)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:06.976654Z","iopub.execute_input":"2021-07-18T11:45:06.977115Z","iopub.status.idle":"2021-07-18T11:45:09.463501Z","shell.execute_reply.started":"2021-07-18T11:45:06.977085Z","shell.execute_reply":"2021-07-18T11:45:09.462411Z"}}
rbf = svm.SVC(kernel='rbf', gamma=0.5, probability = True).fit(xtrain, ytrain)
poly = svm.SVC(kernel='poly', degree=3, probability = True).fit(xtrain, ytrain)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:09.464645Z","iopub.execute_input":"2021-07-18T11:45:09.464899Z","iopub.status.idle":"2021-07-18T11:45:09.842594Z","shell.execute_reply.started":"2021-07-18T11:45:09.464873Z","shell.execute_reply":"2021-07-18T11:45:09.841459Z"}}
poly_pred = poly.predict_proba(xtest)
rbf_pred = rbf.predict_proba(xtest)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:09.843845Z","iopub.execute_input":"2021-07-18T11:45:09.844096Z","iopub.status.idle":"2021-07-18T11:45:09.852187Z","shell.execute_reply.started":"2021-07-18T11:45:09.844070Z","shell.execute_reply":"2021-07-18T11:45:09.851022Z"}}
def getPredictionData(pred):
    probabilities = pandas.Series(pred[pred > 0.2])
    row_indexes = pandas.Series(np.nonzero(pred > 0.2)[0], index = None)
    disease_indexes = pandas.Series(np.nonzero(pred > 0.2)[1], index = None)
    something = pandas.DataFrame(pred)
    prediction_data = pandas.DataFrame([row_indexes, disease_indexes, probabilities]).transpose()
    prediction_data.columns = ['row_index', 'disease_index', 'probability']
    something.columns = data.sort_values('e_prognosis').prognosis.unique()
    prediction_data['disease_name'] = pandas.DataFrame(something.columns).iloc[prediction_data.disease_index].values
    prediction_data = prediction_data.sort_values(['row_index', 'probability'], ascending = False)
    return prediction_data

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:09.855272Z","iopub.execute_input":"2021-07-18T11:45:09.855787Z","iopub.status.idle":"2021-07-18T11:45:09.939782Z","shell.execute_reply.started":"2021-07-18T11:45:09.855752Z","shell.execute_reply":"2021-07-18T11:45:09.938441Z"}}
getPredictionData(rbf_pred)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:09.940992Z","iopub.execute_input":"2021-07-18T11:45:09.941189Z","iopub.status.idle":"2021-07-18T11:45:09.947622Z","shell.execute_reply.started":"2021-07-18T11:45:09.941168Z","shell.execute_reply":"2021-07-18T11:45:09.946752Z"}}
#list(prediction_data[prediction_data.row_index == 704]['disease_name'])

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:09.948432Z","iopub.execute_input":"2021-07-18T11:45:09.948668Z","iopub.status.idle":"2021-07-18T11:45:09.965658Z","shell.execute_reply.started":"2021-07-18T11:45:09.948648Z","shell.execute_reply":"2021-07-18T11:45:09.965009Z"}}
dummy_data = [0]*120
dummy_data += [1]*12
pandas.Series(dummy_data)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:09.967158Z","iopub.execute_input":"2021-07-18T11:45:09.967721Z","iopub.status.idle":"2021-07-18T11:45:09.980635Z","shell.execute_reply.started":"2021-07-18T11:45:09.967690Z","shell.execute_reply":"2021-07-18T11:45:09.979408Z"}}
random.shuffle(dummy_data)
dummy_data = np.array(dummy_data).reshape(1,-1)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:09.982295Z","iopub.execute_input":"2021-07-18T11:45:09.982740Z","iopub.status.idle":"2021-07-18T11:45:09.995025Z","shell.execute_reply.started":"2021-07-18T11:45:09.982707Z","shell.execute_reply":"2021-07-18T11:45:09.993926Z"}}
dummy_data = pca.transform(dummy_data)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:09.996716Z","iopub.execute_input":"2021-07-18T11:45:09.997199Z","iopub.status.idle":"2021-07-18T11:45:10.025794Z","shell.execute_reply.started":"2021-07-18T11:45:09.997150Z","shell.execute_reply":"2021-07-18T11:45:10.024847Z"}}
getPredictionData(rbf.predict_proba(dummy_data))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:10.027242Z","iopub.execute_input":"2021-07-18T11:45:10.027640Z","iopub.status.idle":"2021-07-18T11:45:10.037287Z","shell.execute_reply.started":"2021-07-18T11:45:10.027602Z","shell.execute_reply":"2021-07-18T11:45:10.036176Z"}}
poly_loss = log_loss(ytest, poly_pred)
#poly_accuracy = accuracy_score(ytest, poly_pred)
#poly_f1 = f1_score(ytest, poly_pred, average='weighted')
#print('Accuracy (Polynomial Kernel): ',"%.2f" % (poly_accuracy*100))
print('log loss: ', poly_loss)
#print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2021-07-18T11:45:10.038822Z","iopub.execute_input":"2021-07-18T11:45:10.039087Z","iopub.status.idle":"2021-07-18T11:45:10.053606Z","shell.execute_reply.started":"2021-07-18T11:45:10.039060Z","shell.execute_reply":"2021-07-18T11:45:10.052583Z"}}
#rbf_accuracy = accuracy_score(ytest, rbf_pred)
rbf_loss = log_loss(ytest, rbf_pred)
#rbf_f1 = f1_score(ytest, rbf_pred, average='weighted')
#print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('log loss(RBF): ', rbf_loss)
#print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:45:10.055065Z","iopub.execute_input":"2021-07-18T11:45:10.055394Z","iopub.status.idle":"2021-07-18T11:45:10.083939Z","shell.execute_reply.started":"2021-07-18T11:45:10.055363Z","shell.execute_reply":"2021-07-18T11:45:10.082956Z"}}
pandas.DataFrame(pca.components_, index = ['PC'+str(i) for i in range(1,10)])

# %% [code] {"execution":{"iopub.status.busy":"2021-07-18T11:46:34.770059Z","iopub.execute_input":"2021-07-18T11:46:34.770398Z","iopub.status.idle":"2021-07-18T11:46:34.775922Z","shell.execute_reply.started":"2021-07-18T11:46:34.770369Z","shell.execute_reply":"2021-07-18T11:46:34.774900Z"}}
filename = 'RBF.sav'
pickle.dump(rbf,open(filename, 'wb'))

# %% [code]

