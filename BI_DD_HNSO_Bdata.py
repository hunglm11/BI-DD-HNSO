# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:17:06 2023

@author: Hunglm
"""

'''
Code of paper "Prediction of drug-disease associations based on ensemble meta paths and singular value decomposition".
Please kindly cite the paper:
@article{wu2019EMP-SVD,
  title={Prediction of drug-disease associations based on ensemble meta paths and singular value decomposition},
  author={Wu, Guangsheng and Liu, Juan and Yue, Xiang},
  journal={BMC bioinformatics},
  volume={20},
  number={3},
  pages={134},
  year={2019},
  publisher={BioMed Central}
}
If you have any questions, please do not hesitate to contact wgs@whu.edu.cn
'''

# Python 3 or Anaconda 3 is needed.
import os
import csv
import time
import numpy as np
import pandas as pd
import smote_variants as sv
import random
from sklearn.model_selection import KFold
from numpy import linalg as la
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from scipy import interp
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from datetime import datetime
from datetime import date
from imblearn.over_sampling import RandomOverSampler
from sklearn import metrics
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.svm import SVC
from imblearn.under_sampling import CondensedNearestNeighbour
import datetime
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def calculate_g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    g_mean = (specificity * sensitivity) ** 0.5
    return g_mean

# =============================================================================
# 
# def fbeta_measure(beta, precision, recall):
#     f_beta = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
#     return f_beta
# =============================================================================


def EMP_SVD(drug_disease_matrix, drug_protein_matrix, disease_protein_matrix, k, latent_feature_percent):
    # print(drug_disease_matrix.shape)
    # print(drug_protein_matrix.shape)
    # print(disease_protein_matrix.shape)
    none_zero_position = np.where(drug_disease_matrix != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    ##### code for randomly selected nagative samples
    # zero_position = np.where(drug_disease_matrix == 0)
    # zero_row_index = zero_position[0]
    # zero_col_index = zero_position[1]
    # random.seed(1)
    # zero_random_index = random.sample( range(len(zero_row_index)), len(none_zero_row_index) )
    # zero_row_index = zero_row_index[zero_random_index]
    # zero_col_index = zero_col_index[zero_random_index]
    #drug_protein_dis_matrix = np.dot(drug_protein_matrix, disease_protein_matrix.T) 
    # drug_protein_dis_matrix=np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(drug_protein_matrix, disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T),disease_protein_matrix),disease_protein_matrix.T)
    # drug_protein_dis_matrix=np.dot(drug_protein_matrix, disease_protein_matrix.T)
    #Extract Hight n
    protein_d=np.dot(drug_protein_matrix.T,drug_protein_matrix)#5
    protein_s=np.dot(disease_protein_matrix.T,disease_protein_matrix)
    protein=np.dot(protein_d,protein_s)

    drug_protein_dis_matrix=np.dot(np.dot(drug_protein_matrix, protein),disease_protein_matrix.T)
    
    
    #drug_protein_dis_matrix=np.dot(drug_protein_matrix, disease_protein_matrix.T)
    #drug_protein_dis_matrix=drug_disease_matrix
    zero_deduction_dpd_position = np.where(drug_protein_dis_matrix == 0)
    zero_deduction_dpd_row_index = zero_deduction_dpd_position[0]
    zero_deduction_dpd_col_index = zero_deduction_dpd_position[1]
    # random.seed(1)
    # zero_random_index = random.sample(range(len(zero_deduction_dpd_row_index)), len(none_zero_row_index)*3)
    # zero_row_index = zero_deduction_dpd_row_index[zero_random_index]
    # zero_col_index = zero_deduction_dpd_col_index[zero_random_index]
    # row_index = np.append(none_zero_row_index, zero_row_index)
    # col_index = np.append(none_zero_col_index, zero_col_index)

    row_index = np.append(none_zero_row_index, zero_deduction_dpd_row_index)
    col_index = np.append(none_zero_col_index, zero_deduction_dpd_col_index)
    print(row_index.shape)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    metric_avg = np.zeros((6, 9))
    count = 1
    t = 0
    tprs = []
    precisions = []
    auprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 30)
    mean_recall = np.linspace(0, 1, 30)
    day = datetime.datetime.now()
    day = day.strftime("%Y-%m-%d_%H-%M-%S")
    line = [[day, latent_feature_percent,'', 'Hight Negative', ' ', ' ', '?', '?', '?'],
            [' AUPR', 'AUC', 'PRE', 'REC', 'ACC', 'MCC', 'F1', 'G-Mean', "F_Measure", "f_beta", "PR_AUC"]]
    filename = 'gauSTOMENHigenegativ5.09.01.2025.csv'
    with open(filename, 'a', newline='') as file:

        mywriter = csv.writer(file, delimiter=',')
        for i in line:
            mywriter.writerow(i)
    differing_positions = []
    differing_positions_row = []
    differing_positions_col = []
    baseline_results = []
    for train, test in kf.split(row_index):
        # print('begin cross validation experiment ' + str(count) + '/' + str(kf.n_splits))
        count += 1
        train_drug_disease_matrix = np.copy(drug_disease_matrix)
        test_row = row_index[test]
        test_col = col_index[test]
        train_row = row_index[train]
        train_col = col_index[train]
# =============================================================================
#         np.savetxt('../hunglm/file/test_row' + str(count - 1) + '.txt', test_row, fmt='%d')
#         np.savetxt('../hunglm/file/test_col' + str(count - 1) + '.txt', test_col, fmt='%d')
#         np.savetxt('../hunglm/file/train_row' + str(count - 1) + '.txt', train_row, fmt='%d')
#         np.savetxt('../hunglm/file/train_col' + str(count - 1) + '.txt', train_col, fmt='%d')
# 
# =============================================================================
        train_drug_disease_matrix[test_row, test_col] = 0
# =============================================================================
#         np.savetxt('../hunglm/file/train_drug_disease_matrix_' + str(count - 1) + '.txt', train_drug_disease_matrix,
#                    fmt='%d')
# =============================================================================

        ######################################################################################################
        #### step1: define meta paths

        # meta-path-1: drug->disease
        meta_path_1 = train_drug_disease_matrix

        # meta-path-2: drug->protein->disease
        meta_path_2 = np.dot(drug_protein_matrix, disease_protein_matrix.T)

        # meta-path-3: drug->protein->drug->disease
        meta_path_3 = np.dot(np.dot(drug_protein_matrix, drug_protein_matrix.T), train_drug_disease_matrix)

        # meta-path-4: drug->disease->drug->disease
        meta_path_4 = np.dot(np.dot(train_drug_disease_matrix, train_drug_disease_matrix.T), train_drug_disease_matrix)

        # meta-path-5: drug->disease->protein->disease
        meta_path_5 = np.dot(np.dot(train_drug_disease_matrix, disease_protein_matrix), disease_protein_matrix.T)

        #############################################################################################################
        #### step 2 extract features by SVD
        # latent_feature_percent = 0.03
        (row, col) = train_drug_disease_matrix.shape
        latent_feature_num = int(min(row, col) * latent_feature_percent)

        ## using SVD
        U, Sigma, VT = la.svd(meta_path_1)
        drug_feature_matrix_1 = U[:, :latent_feature_num]
        disease_feature_matrix_1 = VT.T[:, :latent_feature_num]

        U, Sigma, VT = la.svd(meta_path_2)
        drug_feature_matrix_2 = U[:, :latent_feature_num]
        disease_feature_matrix_2 = VT.T[:, :latent_feature_num]

        U, Sigma, VT = la.svd(meta_path_3)
        drug_feature_matrix_3 = U[:, :latent_feature_num]
        disease_feature_matrix_3 = VT.T[:, :latent_feature_num]

        U, Sigma, VT = la.svd(meta_path_4)
        drug_feature_matrix_4 = U[:, :latent_feature_num]
        disease_feature_matrix_4 = VT.T[:, :latent_feature_num]

        U, Sigma, VT = la.svd(meta_path_5)
        drug_feature_matrix_5 = U[:, :latent_feature_num]
        disease_feature_matrix_5 = VT.T[:, :latent_feature_num]

        ##########################################################################################################
        #### step 3: construct training dataset and testing dataset

        train_feature_matrix_1 = []
        train_feature_matrix_2 = []
        train_feature_matrix_3 = []
        train_feature_matrix_4 = []
        train_feature_matrix_5 = []

        train_label_vector = []
        for num in range(len(train_row)):
            feature_vector = np.append(drug_feature_matrix_1[train_row[num], :],
                                       disease_feature_matrix_1[train_col[num], :])
            train_feature_matrix_1.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_2[train_row[num], :],
                                       disease_feature_matrix_2[train_col[num], :])
            train_feature_matrix_2.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_3[train_row[num], :],
                                       disease_feature_matrix_3[train_col[num], :])
            train_feature_matrix_3.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_4[train_row[num], :],
                                       disease_feature_matrix_4[train_col[num], :])
            train_feature_matrix_4.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_5[train_row[num], :],
                                       disease_feature_matrix_5[train_col[num], :])
            train_feature_matrix_5.append(feature_vector)

            train_label_vector.append(drug_disease_matrix[train_row[num], train_col[num]])

        test_feature_matrix_1 = []
        test_feature_matrix_2 = []
        test_feature_matrix_3 = []
        test_feature_matrix_4 = []
        test_feature_matrix_5 = []

        test_label_vector = []
        for num in range(len(test_row)):
            feature_vector = np.append(drug_feature_matrix_1[test_row[num], :],
                                       disease_feature_matrix_1[test_col[num], :])
            test_feature_matrix_1.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_2[test_row[num], :],
                                       disease_feature_matrix_2[test_col[num], :])
            test_feature_matrix_2.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_3[test_row[num], :],
                                       disease_feature_matrix_3[test_col[num], :])
            test_feature_matrix_3.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_4[test_row[num], :],
                                       disease_feature_matrix_4[test_col[num], :])
            test_feature_matrix_4.append(feature_vector)

            feature_vector = np.append(drug_feature_matrix_5[test_row[num], :],
                                       disease_feature_matrix_5[test_col[num], :])
            test_feature_matrix_5.append(feature_vector)

            test_label_vector.append(drug_disease_matrix[test_row[num], test_col[num]])

        train_feature_matrix_1 = np.array(train_feature_matrix_1)
        train_feature_matrix_2 = np.array(train_feature_matrix_2)
        train_feature_matrix_3 = np.array(train_feature_matrix_3)
        train_feature_matrix_4 = np.array(train_feature_matrix_4)
        train_feature_matrix_5 = np.array(train_feature_matrix_5)
        test_feature_matrix_1 = np.array(test_feature_matrix_1)
        test_feature_matrix_2 = np.array(test_feature_matrix_2)
        test_feature_matrix_3 = np.array(test_feature_matrix_3)
        test_feature_matrix_4 = np.array(test_feature_matrix_4)
        test_feature_matrix_5 = np.array(test_feature_matrix_5)
        train_label_vector = np.array(train_label_vector)
        test_label_vector = np.array(test_label_vector)
# =============================================================================
#         print(train_feature_matrix_1.shape)
#         print(train_label_vector.shape, train_label_vector.sum())
#         # =============================================================================
#         #         from imblearn.under_sampling import TomekLinks
#         #         smote = TomekLinks(sampling_strategy='majority')
#         # =============================================================================
#         print("truoc", train_label_vector.shape, train_label_vector.sum())
#         #smote = SMOTE(k_neighbors=15)
#         #smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42)
#         from collections import Counter
#         from sklearn.datasets import make_classification
#         from imblearn.over_sampling import SMOTE
#         from imblearn.under_sampling import RandomUnderSampler
#         from imblearn.pipeline import Pipeline
#         from imblearn.over_sampling import KMeansSMOTE
#         from sklearn.cluster import MiniBatchKMeans
#         from sklearn.datasets import make_blobs
# 
#         from imblearn.over_sampling import SVMSMOTE
#         from imblearn.over_sampling import ADASYN
#         from imblearn.over_sampling import SMOTENC
#         import smote_variants as sv
#         smote = CondensedNearestNeighbour(random_state=42)
#         from imblearn.over_sampling import RandomOverSampler
#         smote = RandomUnderSampler(random_state=42, replacement=True)
#         smote = KMeansSMOTE( kmeans_args={ 'n_clusters': 100 }, smote_args={'k_neighbors': 10})
#         #transform the dataset
#         smote = NearMiss()
#         from imblearn.under_sampling import InstanceHardnessThreshold
#         smote16  = InstanceHardnessThreshold(random_state=42)
#         from imblearn.under_sampling import AllKNN
#         smote17  = AllKNN()
#         from imblearn.under_sampling import CondensedNearestNeighbour
#         smote18  = CondensedNearestNeighbour(random_state=42)
#         from imblearn.combine import SMOTEENN
#         # smote = SMOTEENN(random_state=42)
#         from imblearn.combine import SMOTETomek
# =============================================================================
        # smote = SMOTETomek(random_state=42)
        train_feature_matrix_1, train_label_vector1 = st[k].fit_resample(train_feature_matrix_1, train_label_vector)
        train_feature_matrix_2, train_label_vector2 = st[k].fit_resample(train_feature_matrix_2, train_label_vector)
        train_feature_matrix_3, train_label_vector3 = st[k].fit_resample(train_feature_matrix_3, train_label_vector)
        train_feature_matrix_4, train_label_vector4 = st[k].fit_resample(train_feature_matrix_4, train_label_vector)
        train_feature_matrix_5, train_label_vector5 = st[k].fit_resample(train_feature_matrix_5, train_label_vector)
        print("sau", train_label_vector1.shape, train_label_vector1.sum(), train_label_vector1.shape)
        # print("4",y_train_resampled.sum(), y_train_resampled.shape)
        # transposed_matrix = np.transpose(X_train_resampled)
        #################################################################################################
        #### step 4: training and testing
        # here, using random forest as an example
        clf1 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)
        clf2 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)
        clf3 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)
        clf4 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)
        clf5 = RandomForestClassifier(random_state=1, n_estimators=256, oob_score=True, n_jobs=-1)

        m = test_label_vector.shape[0]  ## the number of test examples
        ensembleScore = np.zeros(m)  ## the ensembled predict_y_proba: average all the predict_y_proba
        ensembleLable = np.zeros(m, dtype=int)
        clf = [clf1, clf2, clf3, clf4, clf5]
        train_feature_matrix = [train_feature_matrix_1, train_feature_matrix_2, train_feature_matrix_3,
                                train_feature_matrix_4, train_feature_matrix_5]
        test_feature_matrix = [test_feature_matrix_1, test_feature_matrix_2, test_feature_matrix_3,
                               test_feature_matrix_4, test_feature_matrix_5]
        train_label_vector = [train_label_vector1, train_label_vector2, train_label_vector3, train_label_vector4,
                              train_label_vector5]
        for i in range(5):

            clf[i].fit(train_feature_matrix[i], train_label_vector[i])
            # print("testing meta-path 1...")
            predict_y_proba = clf[i].predict_proba(test_feature_matrix[i])[:, 1]
            predict_y = clf[i].predict(test_feature_matrix[i])
            # print("evaluating meta-path 1...")
            AUPR = average_precision_score(test_label_vector, predict_y_proba)
            AUC = roc_auc_score(test_label_vector, predict_y_proba)
            recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, predict_y_proba)
            fpr, tpr, thresholds_roc = roc_curve(test_label_vector, predict_y_proba)
            f1 = 2 * (precision * recall) / (precision + recall)
            max_index = np.argwhere(f1 == max(f1))

            PRE = precision[max_index]
            REC = recall[max_index]
            F1 = f1[max_index]
            threshold = thresholds_pr[max_index]
            y_pre = np.copy(predict_y_proba)
            y_pre[y_pre > threshold[0][0]] = 1
            y_pre[y_pre < threshold[0][0]] = 0
            y_pre = y_pre.astype(int)
            ACC = accuracy_score(test_label_vector, y_pre)
            MCC = matthews_corrcoef(test_label_vector, y_pre)
            g_mean = calculate_g_mean(test_label_vector, y_pre)
            #F_Measure = (2 * precision * recall) / (precision + recall)
            #f_beta = fbeta_measure(1.2, precision, recall)
            sorted_indices = np.argsort(recall)
            sorted_recall = recall[sorted_indices]
            sorted_precision = precision[sorted_indices]
            pr_auc = auc(sorted_recall, sorted_precision)
            print(pr_auc, type(pr_auc))
            metric = np.array(
                (AUPR, AUC, PRE[0][0], REC[0][0], ACC, MCC, F1[0][0], g_mean, pr_auc))
            metric_avg[i, :] = metric_avg[i, :] + metric

            for t in range(0, m):
                ensembleScore[t] += predict_y_proba[t]

        ### ensemble all the classifiers built on above meta-paths

        for i in range(0, m):
            ensembleScore[i] = ensembleScore[i] / 5

        AUPR = average_precision_score(test_label_vector, ensembleScore)
        AUC = roc_auc_score(test_label_vector, ensembleScore)
        recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, ensembleScore)
        fpr, tpr, thresholds_roc = roc_curve(test_label_vector, ensembleScore)
        f1 = 2 * (precision * recall) / (precision + recall)
        max_index = np.argwhere(f1 == max(f1))
        PRE = precision[max_index]
        REC = recall[max_index]
        F1 = f1[max_index]
        threshold = thresholds_pr[max_index]
        y_pre = np.copy(ensembleScore)
        y_pre[y_pre > threshold[0][0]] = 1
        y_pre[y_pre < threshold[0][0]] = 0
        y_pre = y_pre.astype(int)
        ACC = accuracy_score(test_label_vector, y_pre)
        MCC = matthews_corrcoef(test_label_vector, y_pre)
        g_mean = calculate_g_mean(test_label_vector, y_pre)
        #F_Measure = (2 * precision * recall) / (precision + recall)
        #f_beta = fbeta_measure(1.2, precision, recall)
        sorted_indices = np.argsort(recall)
        sorted_recall = recall[sorted_indices]
        sorted_precision = precision[sorted_indices]
        pr_auc = auc(sorted_recall, sorted_precision)
        metric = np.array(
            (AUPR, AUC, PRE[0][0], REC[0][0], ACC, MCC, F1[0][0], g_mean, pr_auc))
        metric_avg[5, :] = metric_avg[5, :] + metric

        recall, precision, thresholds_pr = precision_recall_curve(test_label_vector, ensembleScore)
        fpr, tpr, thresholds_roc = roc_curve(test_label_vector, ensembleScore)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        precisions.append(interp(mean_recall, recall, precision))
        auprs.append(AUPR)
        aucs.append(AUC)
        baseline_results.append({'F1-Score': F1[0][0], 'G-mean': g_mean, 'PR-AUC': pr_auc})
              
        roc_auc = metrics.auc(fpr, tpr)
        
    baseline_df = pd.DataFrame(baseline_results)
    baseline_df.to_csv('basic_gausmote_hight_results5.l9.1.csv', index=False)
    print("Baseline results saved to 'baseline_results.csv'")
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = sum(aucs) / 5

    mean_precision = np.mean(precisions, axis=0)
    mean_aupr = sum(auprs) / 5

    print("**********************************************************************************************")
    print(
        "AUPR	    AUC	    PRE	     REC	    ACC	   MCC	   F1    gmeen      PR_AUC")
    print(metric_avg / kf.n_splits)
    print("**********************************************************************************************")
    x = (metric_avg / kf.n_splits)
    b = x.shape[1] * x.shape[0]
    print("b1=", b)
  
    x = np.reshape(x, (6,9))
    with open(filename, 'a', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        # mywriter.writerow(toprint)
        for row in x:
            mywriter.writerow(row)


if __name__ == "__main__":
    print(os.path.abspath('drugDiseaseInteraction.txt'))
    drug_disease_matrix = np.loadtxt("MDABdata/DrugDiseaseMatrix.txt", delimiter='\t', dtype=int)
    print(np.sum(drug_disease_matrix))
    drug_protein_matrix = np.loadtxt('MDABdata/DrugProteinMatrix.txt', delimiter='\t', dtype=int)
    disease_protein_matrix = np.loadtxt('MDABdata/DiseaseProteinMatrix.txt', delimiter='\t', dtype=int)

    # drug_similarity_matrix = np.loadtxt('../hunglm/data/drugSimilarity.txt', delimiter='\t', dtype=float)
    # protein_similarity_matrix = np.loadtxt('..hunglm/data/proteinSimilarity.txt', delimiter='\t', dtype=float)
    # disease_similarity_matrix = np.loadtxt('..hugnlm/data/diseaseSimilarity.txt', delimiter='\t', dtype=float)

    print("Below are the performances of each base classifier and the final ensemble classifier EMP-SVD:")
    print("meta-path-1:drug->disease")
    print("meta-path-2:drug->protein->disease")
    print("meta-path-3:drug->protein->drug->disease")
    print("meta-path-4:drug->disease->drug->disease")
    print("meta-path-5:drug->disease->protein->disease")
    print("ensemble classifier: EMP-SVD")

    # for latent_feature_percent in np.arange(0.01,0.21,0.01):

    latent_feature_percent = 0.05  ## 0.03
    print()
    print()
    print('latent_feature_percent=%s' % (str(latent_feature_percent)))
    # k=26
    start = time.time()
# =============================================================================
#     # while (k < 40):
#     smote1=sv.Gaussian_SMOTE()
#     smote2 = sv.AND_SMOTE()
#     smote3 = sv.SMOTE_D()
#     smote11 = sv.SMOTE_TomekLinks()
#     smote4 = sv.SPY()
#     # smote4 = sv.SMOTE_PSOBAT()
#     smote6 = sv.CURE_SMOTE()
#     smote5 = sv.Random_SMOTE()
#     #smote6 = sv.SMOTE_IPF()
#     #smote7 = sv.Supervised_SMOTE()
#     smote8 = sv.SDSMOTE()
#     smote9 = sv.SMOTEWB()
#     smote10 = sv.kmeans_SMOTE()
#     # smote= sv.SMOTE_D()
#     # smote= sv.SMOTE_D()
#     smote7 = sv.Borderline_SMOTE1()
#     smote12=sv.SMOTE()
#     # smote3 = SVMSMOTE(random_state=101)
#     name = ["SMOTE1", "Gaussian_SMOTE", "AND_SMOTE", "SPY", "SMOTE_D", "Random_SMOTE", "CURE_SMOTE", "SDSMOTE", "SMOTEWB", "kmeans_SMOTE", "SMOTE_TomekLinks","BorderlineSMOTE1"]
#     st = [smote12, smote1, smote2, smote4, smote3, smote5, smote6, smote8, smote9, smote10, smote11,  smote7]
#     # smote= sv.CURE_SMOTE()
#     # smote= sv.SMOTE_PSO()
#     # smote= sv. GASMOTE()
#     # smote= sv.SMOTE_PSOBAT()
#     # smote= sv.Random_SMOTE()
# =============================================================================
    # smote= sv.SMOTE_IPF()
    smote1=sv.Gaussian_SMOTE()
    st=[smote1]
    #latent_feature_percent = 0.3
    k=0
    EMP_SVD(drug_disease_matrix, drug_protein_matrix, disease_protein_matrix, k , latent_feature_percent)
    end = time.time()
    print('Runing time:\t%s\tseconds' % (end - start))

