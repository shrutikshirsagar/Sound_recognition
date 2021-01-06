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
import os
exp_num = [1,2,3,4,5,6]
for num in exp_num:
    
    

    path_out = '/media/amrgaballah/Backup_Plus/Internship_exp/google_audioset_detector_all_results/feat_3_exp/LR/Exp_'+ str(num)+ '/'

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    file_input = '/media/amrgaballah/Backup_Plus/Internship_exp/Exp_'+ str(num)+ '/'+ 'exp'+ str(num)+ '_feat3.mat'

    fileo = file_input.split('/')[-1]
    filen = fileo.split('.')[0]
    print(filen)
    mat = spio.loadmat(file_input)
    random_state = np.random.RandomState(0)
    a=mat['feat_fin']
    x = np.array(a)
    x[x == -inf] = 0
    x[x == +inf] = 0
    x[np.isnan(x)] = 0

    n_classes = 2
    random_state = np.random.RandomState(0)
    output_cols = ['feat', 'Acc']
    out_test = ['Baby_cry', 'carhorn', 'Ding-dong', 'Doorbell','Emergency_vehicle','Firealarm', 'Door_knock', 'Fireengine', 
                'Policecar', 'Ambulance', 'Civildefense_siren', 'Siren', 'Smoke_detector']

    for filename in out_test:
        var1 = 'x'
        feat_3 = ['[:,0:12]','[:,12:24]','[:,24:36]','[:,36:48]','[:, 0:48]', '[:,48:100]', '[:, 100:144]', '[:, 144:188]', '[:,188:276]',  '[:,276:320]', '[:,320:364]','[:,:-1]']
        result_f = np.empty((0,2))

        for j in feat_3:


            print(j)
            X = eval(var1+j)
            print(X.shape)
            Y = x[:,-1]
            n_samples, n_features = X.shape

            s = np.count_nonzero(Y)



            n_samples, n_features = X.shape






#             classifier = svm.SVC()

            classifier = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, l1_ratio=None, max_iter=1000,
                            multi_class='auto', n_jobs=None, penalty='l2',
                            random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                            warm_start=False)

            n_samples, n_features = X.shape


            classifier.fit(X, Y)  
            x_test_ = spio.loadmat('//media/amrgaballah/Backup_Plus/Internship_exp/google_audioset_features/feat_3_final/' + filename + '/' + filename + '.mat')
            a=x_test_['feat_fin']
            x_test = np.array(a)
            x_test[x_test == -inf] = 0
            x_test[x_test == +inf] = 0
            x_test[np.isnan(x_test)] = 0

            var2 = 'x_test'
            X_test = eval(var2+j)


            Y_test  = X_test[:,-1]
            print(Y_test.shape)
            Y_pred=classifier.predict(X_test)

            a = np.sum(Y_pred)
            print(a)
            acc1 = (a/Y_test.shape[0])*100
            print(acc1)
            feat = np.hstack((j, acc1))
            result_f = np.vstack((result_f, feat))

        df=pd.DataFrame(result_f, columns=output_cols)

        df.to_csv(path_out + 'train_exp_'+ str(num) + filename + '_feat3.csv'  ,index=None)

