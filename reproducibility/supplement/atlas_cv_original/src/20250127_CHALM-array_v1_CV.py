from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

class CHALM_array:

    def __init__(self, degree=2, bins=10):
        self.degree = degree
        self.coef = None
        self.bins = bins

    def clculate_kde(self, C):
        kde = KernelDensity(kernel="gaussian",
                            bandwidth=0.05).fit(C.reshape(-1, 1))
        log_dens = kde.score_samples(
            np.linspace(0, 1, self.bins).reshape(-1, 1))
        return np.exp(log_dens)

    def calculate_coef(self, m, c):
        temp = pd.DataFrame([m, c]).dropna(axis=1).T.sort_values(by=0).T.values
        x_d = temp[0]
        y_d = temp[1]
        # when all elements in x is 0, pick the densest y value, otherwise fit a linear function to x~y.
        if all(v == x_d[0] for v in x_d):
            y_pred = np.linspace(0, 1,
                                 self.bins)[self.clculate_kde(y_d).argmax()]
            coef_est = np.array([0] * self.degree + [y_pred])
        else:
            coef_est = np.polyfit(x_d, y_d, self.degree)
        return coef_est

    def fit(self, methylation, chalm):
        if methylation.shape != chalm.shape:
            print("Unmatched sizes of methylation %s and CHALM %s datasets." %
                  (methylation.shape, chalm.shape))
            return
        if not (methylation.index.equals(chalm.index)):
            print("Unmatched CpG IDs of methylation and CHALM datasets.")
            return
        if isinstance(methylation, pd.core.frame.DataFrame) and isinstance(
                chalm, pd.core.frame.DataFrame):
            self.cpg_id = methylation.index.values
            self.sample_id = methylation.columns.values
            methylation = methylation.values
            chalm = chalm.values
        elif isinstance(methylation, np.ndarray) and isinstance(
                chalm, np.ndarray):
            self.cpg_id = np.arange(methylation.shape[0])
            self.sample_id = np.arange(methylation.shape[1])
        self.coef = pd.DataFrame(np.vstack([
            self.calculate_coef(m, c)
            for m, c in tqdm(zip(methylation, chalm), mininterval=30)
        ]),
                                 index=self.cpg_id)
        return self

    def transform_unit(self, cpg, row):
        return np.polyval(self.coef.loc[cpg].values, row)

    def transform(self, methylation_):
        #if self.coef is None:
        #    print('No fitted model.')
        #    return
        #if not (all(np.isin(methylation_.index, self.cpg_id))):
        #    print('Unknown CpGs filter.')
        #    methylation_ = methylation_[np.isin(methylation_.index,
        #                                        self.cpg_id)]
        temp = [
            np.polyval(self.coef.loc[cpg].values, row.values)
            for cpg, row in tqdm(methylation_.iterrows(), mininterval=30)
        ]
        return pd.DataFrame(temp,
                            columns=methylation_.columns,
                            index=methylation_.index)
                            

def cv_chalm_array(methylation, chalm, cv=KFold(n_splits=5), random_state=0, mean=True):
    cv_res = []
    def compute(train_index, test_index):
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        methylation_train, methylation_test = methylation.iloc[:,
                                                               train_index], methylation.iloc[:,
                                                                                              test_index]
        chalm_train, chalm_test = chalm.iloc[:,
                                             train_index], chalm.iloc[:,
                                                                      test_index]
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        chalm_array = CHALM_array(bins=10,
                                  degree=2).fit(methylation_train, chalm_train)
        print("Done fitting", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("Transformation", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        chalm_transformed = chalm_array.transform(methylation_test)
        print("Done transformation", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(chalm_transformed.index.equals(chalm_test.index),
              chalm_transformed.columns.equals(chalm_test.columns))
        if not mean:
            chalm_joined = pd.concat(
                [chalm_test.melt().value,
                 chalm_transformed.melt().value], axis=1)
        else:
            chalm_joined = pd.concat(
                [chalm_test.mean(axis=1),
                 chalm_transformed.mean(axis=1)], axis=1)
        chalm_joined.columns = ['true', 'pred']
        chalm_joined = chalm_joined.dropna(axis=0)
        return chalm_joined
    cv_res = Parallel(n_jobs=-1, backend='loky',verbose=10)(
        delayed(compute)(train_index, test_index)
        for train_index, test_index in cv.split(methylation.T))
    return cv_res


# load data
chalm_matrix = pd.read_pickle('atlas/data/20241023_merged_CHALM_all_sample_beadchip_cpg.pkl')
meth_matrix = pd.read_pickle('atlas/data/20241023_merged_methratio_all_sample_beadchip_cpg.pkl')

chalm_matrix = chalm_matrix.loc[chalm_matrix.isna().sum(
    axis=1) <= 155] # at least 50 valid pairs
meth_matrix = meth_matrix.loc[chalm_matrix.index]

cv_result = cv_chalm_array(meth_matrix,
                           chalm_matrix, cv=RepeatedKFold(n_splits=10, n_repeats=10, random_state=0), mean=True)

#print(cv_result)
pickle.dump(cv_result,open('atlas_cv_original/result/20260127_cv_nature_atlas_205samples_mean.pkl','wb'))