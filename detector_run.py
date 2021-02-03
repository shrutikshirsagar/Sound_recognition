import pandas as pd
import numpy as np
np_load_old = np.load
from sklearn import preprocessing
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k) # set pickle configuration to True
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import auc

from sklearn.model_selection import StratifiedKFold
import scipy.io as spio
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import statistics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn import datasets, metrics, model_selection, svm
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from numpy import inf

file_input = '/media/amrgaballah/Backup_Plus/Internship_exp/Exp_3/exp3_feat2.csv'
path_out = '/media/amrgaballah/Backup_Plus/Internship_exp/Exp_3/'
x = pd.read_csv(file_input, header=0)
x = np.array(x)
print(x)
x[x == -inf] = 0
x[x == +inf] = 0
x[np.isnan(x)] = 0


n_classes = 2


random_state = np.random.RandomState(0)
output_cols = ['feat', 'F1_score', 'Acc', 'recall', 'precision']
cv = StratifiedKFold(n_splits=5)
print('stratified', cv)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)

var1 = 'x'
feat_2 = ['[:, 0:5]', '[:,5:10]', '[:, 10:14]', '[:, 14:27]', '[:,27:39]', '[:,39:75]', '[:,75:81]', '[:,81:-1]', '[:,:-1]']
result_f = np.empty((0,5))
for j in feat_2:
    print('k', var1+j)
  
    
    X = eval(var1+j)
    print(X.shape)
    Y = x[:,-1]
    n_samples, n_features = X.shape
    print(n_classes)
    s = np.count_nonzero(Y)
    print('number of stress examples', s)


    n_samples, n_features = X.shape
    print(n_samples)
    print(n_features)
    acc1 = []
    F1 = []
    precision =[]
    recall = []
    for i, (train, test) in enumerate(cv.split(X, Y)):
        classifier.fit(X[train], Y[train])
        y_pred=classifier.predict(X[test])
        print('x_train number', X[train].shape)
        print('Y_train number', Y[train].shape)

        print('x_test number', X[test].shape)
        print('Y_test number', Y[test].shape)
        sp = np.count_nonzero(Y[test])
        print('number of doorbell examples', sp)

        y_score = classifier.fit(X[train], Y[train]).decision_function(X[test])

        acc=100*sum((y_pred-Y[test])==0)/len(Y[test])
        print('accuracy', acc) 
        acc1.append(acc)
        print(classification_report(Y[test],y_pred))
        print('binary', precision_recall_fscore_support(Y[test],y_pred, average='binary'))
        print('macro', precision_recall_fscore_support(Y[test],y_pred, average='macro'))
        print('micro', precision_recall_fscore_support(Y[test],y_pred, average='micro'))
        print('weighted', precision_recall_fscore_support(Y[test],y_pred, average='weighted'))
        F12 = precision_recall_fscore_support(Y[test],y_pred, average='macro')
        F1.append(F12[2])
        recall.append(F12[1])
        precision.append(F12[0])
        class_names = ['stress','non-stress']
        cm = confusion_matrix(Y[test],y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    F1_f = statistics.mean(F1)
    print('FINAL f1 SCORE', F1_f)
    print(acc1)
    acc_f = statistics.mean(acc1)
    print('Final accuracy', acc_f)
    recall_f = statistics.mean(recall)
    print('recall mean', recall_f)
    precision_f = statistics.mean(precision)
    print('precision mean', precision_f)
    print('j', j)
    print('F1_f', np.array([F1_f]).shape)
    print('acc_f', np.array([acc_f]).shape)
    print('recall_f', np.array([recall_f]).shape)
    print('prec', np.array([precision_f]).shape)
    result = np.hstack((j, np.array([F1_f]), np.array([acc_f]), np.array([recall_f]), np.array([precision_f])))
    
    print(result.shape)
    print(result)
    print(result_f.shape)
    result_f = np.vstack((result_f, result))
    print(result_f)
    print(result_f.shape)
df=pd.DataFrame(result_f,columns=output_cols)
df.to_csv(path_out + 'results_'+ file_input.split('/')[-1]  ,index=None)

        


