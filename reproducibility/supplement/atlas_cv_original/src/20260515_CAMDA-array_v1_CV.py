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

class CAMDA_array:

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

    def fit(self, methylation, camda):
        if methylation.shape != camda.shape:
            print("Unmatched sizes of methylation %s and CAMDA %s datasets." %
                  (methylation.shape, camda.shape))
            return
        if not (methylation.index.equals(camda.index)):
            print("Unmatched CpG IDs of methylation and CAMDA datasets.")
            return
        if isinstance(methylation, pd.core.frame.DataFrame) and isinstance(
                camda, pd.core.frame.DataFrame):
            self.cpg_id = methylation.index.values
            self.sample_id = methylation.columns.values
            methylation = methylation.values
            camda = camda.values
        elif isinstance(methylation, np.ndarray) and isinstance(
                camda, np.ndarray):
            self.cpg_id = np.arange(methylation.shape[0])
            self.sample_id = np.arange(methylation.shape[1])
        self.coef = pd.DataFrame(np.vstack([
            self.calculate_coef(m, c)
            for m, c in tqdm(zip(methylation, camda), mininterval=30)
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


def cv_camda_array(methylation, camda, cv=KFold(n_splits=5), random_state=0, mean=True):
    cv_res = []
    def compute(train_index, test_index):
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
        methylation_train, methylation_test = methylation.iloc[:,
                                                               train_index], methylation.iloc[:,
                                                                                              test_index]
        camda_train, camda_test = camda.iloc[:,
                                             train_index], camda.iloc[:,
                                                                      test_index]
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        camda_array = CAMDA_array(bins=10,
                                  degree=2).fit(methylation_train, camda_train)
        print("Done fitting", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("Transformation", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        camda_transformed = camda_array.transform(methylation_test)
        print("Done transformation", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(camda_transformed.index.equals(camda_test.index),
              camda_transformed.columns.equals(camda_test.columns))
        if not mean:
            camda_joined = pd.concat(
                [camda_test.melt().value,
                 camda_transformed.melt().value], axis=1)
        else:
            camda_joined = pd.concat(
                [camda_test.mean(axis=1),
                 camda_transformed.mean(axis=1)], axis=1)
        camda_joined.columns = ['true', 'pred']
        camda_joined = camda_joined.dropna(axis=0)
        return camda_joined
    cv_res = Parallel(n_jobs=-1, backend='loky',verbose=10)(
        delayed(compute)(train_index, test_index)
        for train_index, test_index in cv.split(methylation.T))
    return cv_res


# load data
# camda_matrix = pd.read_pickle('atlas/data/20241023_merged_CHALM_all_sample_beadchip_cpg.pkl')
meth_matrix = pd.read_pickle('atlas/data/20241023_merged_methratio_all_sample_beadchip_cpg.pkl')
camda_matrix = pd.read_pickle(
    "atlas/data/20260514_merged_CAMDA_all_sample_beadchip_cpg.pkl"
)

camda_matrix = camda_matrix.loc[camda_matrix.isna().sum(
    axis=1) <= 155] # at least 50 valid pairs
meth_matrix = meth_matrix.loc[camda_matrix.index]

cv_result = cv_camda_array(meth_matrix,
                           camda_matrix, cv=RepeatedKFold(n_splits=10, n_repeats=10, random_state=0), mean=True)

#print(cv_result)
pickle.dump(cv_result,open('atlas_cv_original/result/20260516_cv_nature_atlas_205samples_mean_camda.pkl','wb'))