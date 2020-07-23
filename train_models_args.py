#!/usr/bin/env python

#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 8      # cores requested
#SBATCH --mem=24000  # memory in Mb
#SBATCH -t 140:00:00  # time requested in hour:minute:second


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


import pandas as pd
import sklearn as sk 
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ShuffleSplit
import os
import numpy as np 
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score, precision_score, matthews_corrcoef, f1_score
from joblib.parallel import Parallel
import itertools 
import joblib
from joblib.parallel import Parallel, delayed
import argparse





import numpy as np
import scipy.sparse


def linearkernel(data_1, data_2):
    return np.dot(data_1, data_2.T)


def tanimotokernel(data_1, data_2):
    if isinstance(data_1, scipy.sparse.csr_matrix) and isinstance(data_2, scipy.sparse.csr_matrix):
        return _sparse_tanimotokernel(data_1, data_2)
    elif isinstance(data_1, scipy.sparse.csr_matrix) or isinstance(data_2, scipy.sparse.csr_matrix):
        # try to sparsify the input
        return _sparse_tanimotokernel(scipy.sparse.csr_matrix(data_1), scipy.sparse.csr_matrix(data_2))
    else:  # both are dense
        return _dense_tanimotokernel(data_1, data_2)
    
    
    
    
def _dense_tanimotokernel(data_1, data_2):
    """
    Tanimoto kernel
        K(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)
    as defined in:
    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    https://www.sciencedirect.com/science/article/pii/S0893608005001693
    http://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Ralaivola2005Graph.pdf
    """

    norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
    norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
    prod = data_1.dot(data_2.T)

    divisor = (norm_1 + norm_2.T - prod) + np.finfo(data_1.dtype).eps
    return prod / divisor



def _sparse_tanimotokernel(data_1, data_2):
    """
    Tanimoto kernel
        K(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)
    as defined in:
    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    https://www.sciencedirect.com/science/article/pii/S0893608005001693
    http://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Ralaivola2005Graph.pdf
    """

    norm_1 = np.array(data_1.power(2).sum(axis=1).reshape(data_1.shape[0], 1))
    norm_2 = np.array(data_2.power(2).sum(axis=1).reshape(data_2.shape[0], 1))
    prod = data_1.dot(data_2.T).A

    divisor = (norm_1 + norm_2.T - prod) + np.finfo(data_1.dtype).eps
    result = prod / divisor
    return result


def _minmaxkernel_numpy(data_1, data_2):
    """
    MinMax kernel
        K(x, y) = SUM_i min(x_i, y_i) / SUM_i max(x_i, y_i)
    bounded by [0,1] as defined in:
    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.483&rep=rep1&type=pdf
    """
    return np.stack([(np.minimum(data_1, data_2[cpd,:]).sum(axis=1) / np.maximum(data_1, data_2[cpd,:]).sum(axis=1))  for cpd in range(data_2.shape[0])],axis=1)


try: 
    import numba
    from numba import njit, prange
    

    @njit(parallel=True,fastmath=True)
    def _minmaxkernel_numba(data_1, data_2):
        """
        MinMax kernel
            K(x, y) = SUM_i min(x_i, y_i) / SUM_i max(x_i, y_i)
        bounded by [0,1] as defined in:
        "Graph Kernels for Chemical Informatics"
        Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
        Neural Networks
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.92.483&rep=rep1&type=pdf
        """


        result = np.zeros((data_1.shape[0], data_2.shape[0]), dtype=np.float64)

        for i in prange(data_1.shape[0]):
            for j in prange(data_2.shape[0]):
                result[i,j] = _minmax_two_fp(data_1[i], data_2[j])
        return result


    @njit(fastmath=True)
    def _minmax_two_fp(fp1, fp2):
        common = numba.int32(0)
        maxnum = numba.int32(0)
        i = 0
        
        while i < len(fp1):
            min_ = fp1[i]
            max_ = fp2[i]

            if min_ > max_:
                min_ = fp2[i]
                max_ = fp1[i]

            common += min_
            maxnum += max_

            i += 1
            
        return numba.float64(common) / numba.float64(maxnum)

except:
    
    print("Couldn't find numba. I suggest to install numba to compute the minmax kernel much much faster")
    
    minmaxkernel = _minmaxkernel_numpy



def precompute_gram(target, kernel):
    df = pd.read_pickle("{}_df.pkl.gz".format(target))
    training = df[df.trainingset_class == "training"]
    test = df[df.trainingset_class == "test"]
    validation = df[df.trainingset_class == "validation"]

    dtype = np.int32 if kernel == "minmax" else np.float64
    
    training_X = np.array([np.array(e) for e in training.cfp.values],dtype=dtype)
    training_Y = np.array(training.activity_label, dtype=np.float64)

    if kernel =="tanimoto":
        gram_matrix_training = tanimotokernel(training_X,training_X)
    elif kernel =='minmax':
        gram_matrix_training = minmaxkernel(training_X,training_X)
    else:
        gram_matrix_training = linearkernel(training_X,training_X)
                
    np.save("{}_{}_training_X.npy".format(target, kernel),gram_matrix_training)
    np.save("{}_{}_training_Y.npy".format(target, kernel),training_Y)
    del gram_matrix_training

    test_X = np.array([np.array(e) for e in test.cfp.values],dtype=dtype)
    test_Y = np.array(test.activity_label, dtype=np.float64)
            
    if kernel =="tanimoto":
        gram_matrix_test = tanimotokernel(test_X, training_X)
    elif kernel =='minmax':
        gram_matrix_test = minmaxkernel(test_X,training_X)
    else:
        gram_matrix_test = linearkernel(test_X, training_X)

    np.save("{}_{}_test_X.npy".format(target, kernel),gram_matrix_test)
    np.save("{}_{}_test_Y.npy".format(target, kernel),test_Y)
    del gram_matrix_test

    validation_X = np.array([np.array(e) for e in validation.cfp.values],dtype=dtype)
    validation_Y = np.array(validation.activity_label, dtype=np.float64)

    if kernel =="tanimoto":
        gram_matrix_validation = tanimotokernel(validation_X, training_X)
    elif kernel =='minmax':
        gram_matrix_validation = minmaxkernel(validation_X,training_X)
    else:
        gram_matrix_validation = linearkernel(validation_X, training_X)
                
    np.save("{}_{}_validation_X.npy".format(target, kernel),gram_matrix_validation)
    np.save("{}_{}_validation_Y.npy".format(target, kernel),validation_Y)
    del gram_matrix_validation
      

def train_and_evaluate(target,c,kernel, bal, mmap_mode=None):

    if mmap_mode == "None":
        mmap_mode = None

    print("Start {} {} {} {}".format(target,c,kernel, bal))
    if os.path.exists('models/{}_c_{}_kernel_{}_{}_proba.pkl'.format(target, c,kernel,bal)):
        return
    
    if not os.path.exists("{}_{}_validation_Y.npy".format(target, kernel)):            
        precompute_gram(target, kernel)
    
    training_X = np.load("{}_{}_training_X.npy".format(target, kernel), mmap_mode=mmap_mode)
    training_Y = np.ascontiguousarray(np.load("{}_{}_training_Y.npy".format(target, kernel)), dtype=np.float64) 
    
    test_X = np.load("{}_{}_test_X.npy".format(target, kernel), mmap_mode=mmap_mode)
    test_Y = np.ascontiguousarray(np.load("{}_{}_test_Y.npy".format(target, kernel)), dtype=np.float64) 
                     
    validation_X = np.load("{}_{}_validation_X.npy".format(target, kernel), mmap_mode=mmap_mode)
    validation_Y = np.ascontiguousarray(np.load("{}_{}_validation_Y.npy".format(target, kernel)), dtype=np.float64) 

    
    def advertize_mmap(arr, advise):
        
        #MADV_NORMAL       0
        #MADV_RANDOM       1
        #MADV_WILLNEED     3
        #MADV_DONTNEED     4
        
        if mmap_mode:    
            #numpy is quite slow with random access mmap. lets give it some hints for random access
            import platform
            if platform.system() == "Linux" or "Darwin":
                import ctypes
                libc = "libc.so.6" if platform.system() == "Linux" else "libc.dylib"
                madvise = ctypes.CDLL(libc).madvise
                madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                madvise.restype = ctypes.c_int
                madvise(arr.ctypes.data, arr.size * arr.dtype.itemsize, advise)
                #assert madvise(arr.ctypes.data, arr.size * arr.dtype.itemsize, advise) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM
    
    def score_clf(clf):
        sets = {"Training set": (training_X, training_Y) ,"Test set": (test_X, test_Y), "Validation set": (validation_X, validation_Y) }
        scores_binary = {"Balanced Accuracy": lambda x,y: balanced_accuracy_score(x,y, adjusted=False), "Recall": recall_score, "Precision":  precision_score, "MCC": matthews_corrcoef, "F1": f1_score }
        scores_proba = {"ROC AUC": roc_auc_score }

        scores = {}
        for setname, data in sets.items():
            data_X = data[0]
            data_Y = data[1]
            
            advertize_mmap(data_X, 0)
            predicted_Y = clf.predict(data_X)
            predicted_Y_proba = clf.predict_proba(data_X)[:,1]
            advertize_mmap(data_X, 4)
            for scorename, score_fn in scores_binary.items():
                scores["{}\t{}".format(setname, scorename)] = score_fn(data_Y, predicted_Y)
            
            for threshold in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
                for scorename, score_fn in scores_binary.items():
                    scores["{}\t{}\tthreshold_{}".format(setname, scorename, threshold)] = score_fn(data_Y, np.array(predicted_Y_proba >= threshold, dtype=np.int))

            for scorename, score_fn in scores_proba.items():
                scores["{}\t{}".format(setname, scorename)] = score_fn(data_Y, predicted_Y_proba)

        return scores

    
    def build_test_svm(c, kernel, bal, training_X, training_Y, test_X, test_Y, validation_X, validation_Y):
        
        advertize_mmap(training_X, 0)
        if bal == "unbalanced":
            class_weight = None
        else:
            class_weight = 'balanced'
        clf = svm.SVC(C=c, random_state=1234, kernel='precomputed', cache_size=1900, probability=True, class_weight=class_weight)
        
        clf.fit(training_X, training_Y)

        scores = score_clf(clf)
        
        return clf, scores


    if not os.path.exists('models/{}_c_{}_kernel_{}_{}_proba.pkl'.format(target, c,kernel,bal)):
        clf, scores = build_test_svm(c,kernel, bal, training_X, training_Y,test_X, test_Y, validation_X,validation_Y)
        joblib.dump(clf, 'models/{}_c_{}_kernel_{}_{}_proba.pkl'.format(target, c,kernel,bal)) 
        with open('models/{}_c_{}_kernel_{}_{}_proba.stats'.format(target, c,kernel,bal), "w") as fd:
            for scorename, score in scores.items():
                line = "{}:\t{}\n".format(scorename, score)
                fd.write(line)
    else:
        clf = joblib.load('models/{}_c_{}_kernel_{}_{}_proba.pkl'.format(target, c,kernel,bal))
        scores = score_clf(clf)
        with open('models/{}_c_{}_kernel_{}_{}_proba.stats'.format(target, c,kernel,bal), "w") as fd:
            for scorename, score in scores.items():
                line = "{}:\t{}\n".format(scorename, score)
                fd.write(line)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute gram matrix')
    parser.add_argument('target', type=str, help='Target dataset to lead')
    parser.add_argument('c', type=float, help='C parameter of SVM')
    parser.add_argument('k', type=str, help='Kernel of SVM')
    parser.add_argument('bal', type=str, help='Balance Classes')
    parser.add_argument('mmap', type=str, help='MMAP mode for loading the Gram Matrix')
    args = parser.parse_args()
    
    
    if not os.path.exists("{}_{}_validation_Y.npy".format(args.target, args.k)):            
        precompute_gram(target, kernel)
    train_and_evaluate(args.target, args.c, args.k, args.bal, mmap_mode=args.mmap)
    
    
    
    
#targets = ["DRD2"]
#cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10 ,100, 1000, 10000]
#gs = [0.00001,0.0001,0.001,0.01,0.1,1,10,100, 1000, 10000]

#combinations = list(itertools.product(targets, cs, gs))[::-1]
#results_training = {}
#results_test = {}

#results = Parallel()(delayed(train_and_evaluate)(target, c, g) for target,c,g in combinations)
# DRD2_nm_c_5000.0_kernel_1e-05_k_rbf.stats:Test set adjusted balanced accuracy:       0.4300765325077398
