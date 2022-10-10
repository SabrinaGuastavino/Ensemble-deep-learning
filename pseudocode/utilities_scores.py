#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:38:30 2022

@author: sabry
"""
import numpy
from sklearn.metrics import confusion_matrix


# UTILITIES

#
def compute_weight_cm_tss_threshold(y, pred,threshold):
    # Compute the value-weighted confusion matrix and value-weighted skill scores (i.e., wTSS, wHSS and wCSI) given in input:
    # - the actual label vector y
    # - the predicted vector y_pred representing probabilities 
    # - the threshold which converts the probabilities in y_pred into binary outcomes
    
    pred_threshold = pred > threshold
    
    weighted_cm, tss, hss, CSI = compute_weight_cm_tss(y, pred_threshold)
  
    return weighted_cm, tss, hss, CSI
        
#
def compute_cm_tss_threshold(y, pred,threshold):
    # Compute confusion matrix and skill scores (i.e. TSS, HSS and CSI) given in input
    # - the actual label vector y 
    # - the predicted vector y_pred representing probabilities
    # - the threshold which converts the probabilities in y_pred into binary outcomes
    pred_threshold = pred > threshold
    cm, tss, hss, CSI = compute_cm_tss(y, pred_threshold)
    
    return cm, tss, hss, CSI
        
    
#
def compute_cm_tss(y, pred):
    # Compute confusion matrix and skill scores (i.e. TSS, HSS and CSI) given in input
    # - the actual label vector y 
    # - the predicted vector pred representing binary outcomes

    cm = confusion_matrix(y,pred)
    if cm.shape[0] == 1 and sum(y_true) == 0:
        a = 0.
        d = float(cm[0, 0])
        b = 0.
        c = 0.
    elif cm.shape[0] == 1 and sum(y_true) == y_true.shape[0]:
        a = float(cm[0, 0])
        d = 0.
        b = 0.
        c = 0.
    elif cm.shape[0] == 2:
        a = float(cm[1, 1])
        d = float(cm[0, 0])
        b = float(cm[0, 1])
        c = float(cm[1, 0])
    TP = a
    TN = d
    FP = b
    FN = c

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2
    
    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))
    
    if TP+FP+FN==0:
        CSI = 0
    else:
        CSI = TP/(TP+FP+FN)

    
    return cm, tss, hss, CSI

#
def compute_weight_cm_tss(y, pred):
    # Compute the value-weighted confusion matrix and value-weighted skill scores (i.e., wTSS, wHSS and wCSI) given in input:
    # - the actual label vector y
    # - the predicted vector pred representing binary outcomes

    TN,FP,FN,TP = weighted_confusion_matrix(y,pred)

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2
    
    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))
    
    if TP+FP+FN==0:
        CSI=0
    else:
        CSI = TP/(TP+FP+FN)
    
    weighted_cm = numpy.zeros((2,2))
    weighted_cm[0,0]=TN
    weighted_cm[0,1]=FP
    weighted_cm[1,0]=FN
    weighted_cm[1,1]=TP
    
    return weighted_cm, tss, hss, CSI

#
def optimize_threshold_skill_scores_weight_matrix(probability_prediction, Y_training):
# computation of the best thresholds by optimizing value-weighted skill scores given in input
# - the vector of the predicted probabilities probability_prediction (on the training set)
# - the actual label vector Y_training 
# the function returns 
# - best_xss_threshold : the optimum threshold computed with respect to the best wNSS
# - metrics_training : the value-weighted CM computed with the optimum threshold with respect to the best wNSS
# - nss_vector : the vector of wNSS computed for a set of values of thresholds
# - best_xss_threshold_tss : the optimum threshold computed with respect to the best wTSS 
# - best_xss_threshold_hss : the optimum threshold computed with respect to the best wHSS  
# - best_xss_threshold_csi : the optimum threshold computed with respect to the best wCSI
# - best_xss_threshold_tss_hss : the optimum threshold computed with respect to the best (wTSS+wHSS)/2

    n_samples = 100
    step = 1. / n_samples
    
    xss = -1.
    xss_threshold = 0
    Y_best_predicted = numpy.zeros((Y_training.shape))
    tss_vector = numpy.zeros(n_samples)
    hss_vector = numpy.zeros(n_samples)
    csi_vector = numpy.zeros(n_samples)
    
    xss_threshold_vector = numpy.zeros(n_samples)
    a = probability_prediction.max()
    b = probability_prediction.min()
    for threshold in range(1, n_samples):
        xss_threshold = step * threshold * numpy.abs(a - b) + b
        xss_threshold_vector[threshold] = xss_threshold
        Y_predicted = probability_prediction > xss_threshold
        res = metrics_classification_weight(Y_training > 0, Y_predicted, print_skills=False)
        tss_vector[threshold] = res['tss']
        hss_vector[threshold] = res['hss']
        csi_vector[threshold] = res['csi']
            
    nss_vector = 0.5*((tss_vector/numpy.max(tss_vector)) + (hss_vector/numpy.max(hss_vector)))
    idx_best_nss = numpy.where(nss_vector==numpy.max(nss_vector))  
    print('idx best wnss=',idx_best_nss)
    #best wNSS
    best_xss_threshold = xss_threshold_vector[idx_best_nss]
    if len(best_xss_threshold)>1:
        best_xss_threshold = best_xss_threshold[0]
        Y_best_predicted = probability_prediction > best_xss_threshold
    else:
        Y_best_predicted = probability_prediction > best_xss_threshold
    print('best wNSS')
    metrics_training = metrics_classification_weight(Y_training > 0, Y_best_predicted)
    
    #best wTSS
    idx_best_tss = numpy.where(tss_vector==numpy.max(tss_vector))  
    print('idx best wtss=',idx_best_tss)
    best_xss_threshold_tss = xss_threshold_vector[idx_best_tss]
    if len(best_xss_threshold_tss)>1:
        best_xss_threshold_tss = best_xss_threshold_tss[0]
        Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
    else:
        Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
    print('best wTSS')
    metrics_training_tss = metrics_classification_weight(Y_training > 0, Y_best_predicted_tss)
    
    #best wHSS
    idx_best_hss = numpy.where(hss_vector==numpy.max(hss_vector))  
    print('idx best whss=',idx_best_hss)
    best_xss_threshold_hss = xss_threshold_vector[idx_best_hss]
    if len(best_xss_threshold_hss)>1:
        best_xss_threshold_hss = best_xss_threshold_hss[0]
        Y_best_predicted_hss = probability_prediction > best_xss_threshold_hss
    else:
        Y_best_predicted_hss = probability_prediction > best_xss_threshold_hss
    print('best wHSS')
    metrics_training_hss = metrics_classification_weight(Y_training > 0, Y_best_predicted_hss)
    
    #best wCSI
    idx_best_csi = numpy.where(csi_vector==numpy.max(csi_vector))  
    print('idx best wcsi=',idx_best_csi)
    best_xss_threshold_csi = xss_threshold_vector[idx_best_csi]
    if len(best_xss_threshold_csi)>1:
        best_xss_threshold_csi = best_xss_threshold_csi[0]
        Y_best_predicted_csi = probability_prediction > best_xss_threshold_csi
    else:
        Y_best_predicted_csi = probability_prediction > best_xss_threshold_csi
    print('best CSI')
    metrics_training_csi = metrics_classification_weight(Y_training > 0, Y_best_predicted_csi)
    
    #best (wTSS+wHSS)/2
    comb_tss_hss = (hss_vector+tss_vector)/2.
    idx_best_tss_hss = numpy.where(comb_tss_hss==numpy.max(comb_tss_hss)) 
    print('idx best (wtss+whss)/2 =',idx_best_tss_hss)
    best_xss_threshold_tss_hss = xss_threshold_vector[idx_best_tss_hss]
    if len(best_xss_threshold_tss_hss)>1:
        best_xss_threshold_tss_hss = best_xss_threshold_tss_hss[0]
        Y_best_predicted_tss_hss = probability_prediction > best_xss_threshold_tss_hss
    else:
        Y_best_predicted_tss_hss = probability_prediction > best_xss_threshold_tss_hss
    print('best (wTSS+wHSS)/2')
    metrics_training_tss_hss = metrics_classification_weight(Y_training > 0, Y_best_predicted_tss_hss)
    
    

    return best_xss_threshold, metrics_training, nss_vector, best_xss_threshold_tss, best_xss_threshold_hss, best_xss_threshold_csi,best_xss_threshold_tss_hss


#
def metrics_classification_weight(y_real, y_pred, print_skills=True):
# computation of value-weighted confusion matrix and value-weighted skill scores (wfar, wpod, wacc, whss, wtss, wfnfp, wcsi) given in input:
# - y_real :  the actual label vector
# - y_pred :  the vector of predicted probabilities

    cm, far, pod, acc, hss, tss, fnfp, csi = classification_skills_weight(y_real, y_pred)

    if print_skills:
        print ('confusion matrix')
        print (cm)
        print ('false alarm ratio       \t', far)
        print ('probability of detection\t', pod)
        print ('accuracy                \t', acc)
        print ('hss                     \t', hss)
        print ('tss                     \t', tss)
        print ('balance                 \t', fnfp)
        print ('csi                 \t', csi)

    balance_label = float(sum(y_real)) / y_real.shape[0]


    return {
        "cm": cm,
        "far": far,
        "pod": pod,
        "acc": acc,
        "hss": hss,
        "tss": tss,
        "fnfp": fnfp,
        "balance label": balance_label,
        "csi": csi}

def classification_skills_weight(y_real, y_pred):
# computation of value-weighted confusion matrix and value-weighted skill scores (wfar, wpod, wacc, whss, wtss, wfnfp, wcsi) given in input:
# - y_real :  the actual label vector
# - y_pred :  the vector of predicted probabilities

    TN,FP,FN,TP = weighted_confusion_matrix(y_real, y_pred)

    if (TP + FP + FN + TN) == 0.:
        if (TP + TN) == 0.:
            acc = 0.  # float('NaN')
        else:
            acc = -100  # float('Inf')
    else:
        acc = (TP + TN) / (TP + FP + FN + TN)

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2

    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))

    if FP == 0.:
        if FN == 0.:
            fnfp = 0.  # float('NaN')
        else:
            fnfp = -100  # float('Inf')
    else:
        fnfp = FN / FP

    if (TP + FN) == 0.:
        if TP == 0.:
            pod = 0  # float('NaN')
        else:
            pod = -100  # float('Inf')
    else:
        pod = TP / (TP + FN)


    if (TP + FP) == 0.:
        if FP == 0.:
            far = 0.  # float('NaN')
        else:
            far = -100  # float('Inf')
    else:
        far = FP / (TP + FP)

    #acc = (a + d) / (a + b + c + d)
    #tpr = a / (a + b)
    #tnr = d / (d + c)
    #wtpr = a / (a + b) * (a + c) / (a + b + c + d) + d / (c + d) * (b + d) / (a + b + c + d)
    #pacc = a / (a + c)
    #nacc = d / (b + d)
    #wacc = a / (a + c) * (a + c) / (a + b + c + d) + d / (b + d) * (b + d) / (a + b + c + d)

    # if the cm has a row or a column equal to 0, we have bad tss
    if TP+FN == 0 or TN+FP == 0 or TP+FP == 0 or TN+FN == 0:
        tss = 0
        
    if TP+FP+FN==0:
        csi = 0
    else:
        csi = TP/(TP+FP+FN)
    
    
    weighted_cm = numpy.zeros((2,2))
    weighted_cm[0,0]=TN
    weighted_cm[0,1]=FP
    weighted_cm[1,0]=FN
    weighted_cm[1,1]=TP
    #, pod, acc, hss, tss, fnfp, tpr, tnr, pacc, nacc, wacc, wtpr)
    return weighted_cm, far, pod, acc, hss, tss, fnfp, csi

#
def weighted_confusion_matrix(y_true, y_pred):
# Computation of the value-weighted confusion matrix given in input:
# - y_true : the actual label vector
# - y_pred : the predicted binary vector 
    TP_values = numpy.logical_and(numpy.equal(y_true, True), numpy.equal(y_pred, True))
    idx_TP=numpy.where(TP_values==True)
    TP=len(idx_TP[0])
    
    TN_values = numpy.logical_and(numpy.equal(y_true, False), numpy.equal(y_pred, False))
    idx_TN=numpy.where(TN_values==True)
    TN=len(idx_TN[0])


    FP_values = numpy.logical_and(numpy.equal(y_true, False), numpy.equal(y_pred, True))
    idx_FP=numpy.where(FP_values==True)
    mask = [1./2.,1./3.,1./4.]
    FP=0
    window_hour=3
    if y_true.shape[0] >=6:
        
        for t in idx_FP[0]: 
            if t >=  window_hour and t <= len(y_true)- window_hour-1:
                #window -4 +4
                y_true_window = y_true[t- window_hour:t+ window_hour+1]
                y_pred_window = y_pred[t- window_hour:t+ window_hour+1]
        
                if len(numpy.where(y_true_window==1)[0]) >= 1:
                    count_FP = 1-numpy.max(mask*y_true[t+1:t+ window_hour+1])
                else:
                    count_FP = 2
            elif t<window_hour:
                y_true_window = y_true[:t+ window_hour+1]
                y_pred_window = y_pred[:t+ window_hour+1]
                if len(numpy.where(y_true_window==1)[0]) >= 1:
                    count_FP = 1-numpy.max(mask*y_true[t+1:t+ window_hour+1])
                else:
                    count_FP = 2
            elif t > len(y_true)-window_hour-1:
                y_true_window = y_true[t- window_hour:]
                y_pred_window = y_pred[t- window_hour:]
                if len(numpy.where(y_true_window==1)[0]) >= 1:
                    if t < len(y_true)-1:
                        count_FP = 1-numpy.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                    elif t == len(y_true)-1:
                        count_FP = 1
                
                else:
                    count_FP = 2
            FP=FP+count_FP
    
    if y_true.shape[0]<6:
        for t in idx_FP[0]: 
            if t<window_hour:
                if t+window_hour+1<=numpy.shape(y_true)[0]:
                    y_true_window = y_true[:t+ window_hour+1]
                    y_pred_window = y_pred[:t+ window_hour+1]
                    if len(numpy.where(y_true_window==1)[0]) >= 1:
                        count_FP = 1-numpy.max(mask*y_true[t+1:t+ window_hour+1])
                    else:
                        count_FP = 2
                elif t+window_hour+1 >numpy.shape(y_true)[0]:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(numpy.where(y_true_window==1)[0]) >= 1:
                        if t < len(y_true)-1:
                            count_FP = 1-numpy.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                        elif t == len(y_true)-1:
                            count_FP = 1
                
                    else:
                        count_FP = 2
            elif t > len(y_true)-window_hour-1:
                if t- window_hour>=0:
                    y_true_window = y_true[t- window_hour:]
                    y_pred_window = y_pred[t- window_hour:]
                    if len(numpy.where(y_true_window==1)[0]) >= 1: 
                        if t < len(y_true)-1:
                            count_FP = 1-numpy.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                        elif t == len(y_true)-1:
                            count_FP = 1
                
                    else:
                        count_FP = 2
                elif t- window_hour<0:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(numpy.where(y_true_window==1)[0]) >= 1:
                        if t < len(y_true)-1:
                            count_FP = 1-numpy.max(mask[0:len(y_true)-t-1]*y_true[t+1:])
                        elif t == len(y_true)-1:
                            count_FP = 1
                
                    else:
                        count_FP = 2
            FP=FP+count_FP
                        
        
    FN_values = numpy.logical_and(numpy.equal(y_true, True), numpy.equal(y_pred, False))
    idx_FN=numpy.where(FN_values==True)
    FN=0
    mask_FN=[1./4.,1./3.,1./2.]
    if y_true.shape[0]>=6:
       
        for t in idx_FN[0]: 
            
            if t >=  window_hour and t <= len(y_true)- window_hour-1:
             #window -4 +4
                y_true_window = y_true[t- window_hour:t+ window_hour+1]
                y_pred_window = y_pred[t- window_hour:t+ window_hour+1]
                if len(numpy.where(y_pred_window==1)[0]) >= 1:
                    count_FN = 1-numpy.max(mask_FN*y_pred[t- window_hour:t])
                else:
                    count_FN = 2
            elif t<window_hour:
                y_true_window = y_true[:t+ window_hour+1]
                y_pred_window = y_pred[:t+ window_hour+1]
                if len(numpy.where(y_pred_window==1)[0]) >= 1:
                    if t > 0:
                        count_FN = 1-numpy.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])#
                    elif t ==0:
                        count_FN = 1
                else:
                    count_FN = 2
            elif t > len(y_true)- window_hour-1:
                y_true_window = y_true[t- window_hour:]
                y_pred_window = y_pred[t- window_hour:]
                if len(numpy.where(y_pred_window==1)[0]) >= 1:
                    count_FN = 1-numpy.max(mask_FN*y_pred[t- window_hour:t])
                else:
                    count_FN = 2
       
            FN=FN+count_FN
                
                
    if y_true.shape[0]<6:
        for t in idx_FN[0]:
            if t<window_hour:
                if t+window_hour+1<=numpy.shape(y_true)[0]:
                    y_true_window = y_true[:t+ window_hour+1]
                    y_pred_window = y_pred[:t+ window_hour+1]
                    if len(numpy.where(y_pred_window==1)[0]) >= 1:
                        if t > 0:
                            count_FN = 1-numpy.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])#
                        elif t ==0:
                            count_FN = 1
                    else:
                        count_FN = 2
                elif t+window_hour+1 >numpy.shape(y_true)[0]:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(numpy.where(y_pred_window==1)[0]) >= 1:
                        if t > 0:
                            count_FN = 1-numpy.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])
                        elif t == 0:
                            count_FN = 1
                    else:
                        count_FN = 2
                            
            elif t > len(y_true)-window_hour-1:
                if t- window_hour>=0:
                    y_true_window = y_true[t- window_hour:]
                    y_pred_window = y_pred[t- window_hour:]
                    if len(numpy.where(y_pred_window==1)[0]) >= 1:
                        count_FN = 1-numpy.max(mask_FN*y_pred[t- window_hour:t])
                    else:
                        count_FN = 2
                elif t- window_hour<0:
                    y_true_window = y_true
                    y_pred_window = y_pred
                    if len(numpy.where(y_pred_window==1)[0]) >= 1:
                        if t > 0:
                            count_FN = 1-numpy.max(mask_FN[window_hour-t:window_hour]*y_pred[:t])#
                        elif t == 0:
                            count_FN = 1
                    else:
                        count_FN = 2
        
            FN=FN+count_FN
            
            
    return TN, FP, FN, TP

#
def weighted_confusion_matrix_threshold(y_true, y_pred, threshold):
# Computation of the value-weighted confusion matrix given in input:
# - y_true :  the actual label vector
# - y_pred : the vector of predicted probabilities
# - threshold : the threshold which convert the probabilities in binary outcomes
    y_pred = y_pred>threshold
    TN,FP,FN,TP = weighted_confusion_matrix(y_true, y_pred)
    return TN,FP,FN,TP 


#
def optimize_threshold_skill_scores(probability_prediction, Y_training):
# computation of the best thresholds by optimizing skill scores given in input
# - the vector of the predicted probabilities probability_prediction (on the training set)
# - the actual label vector Y_training 
# the function returns 
# - best_xss_threshold : the optimum threshold computed with respect to the best NSS
# - metrics_training : the value-weighted CM computed with the optimum threshold with respect to the best NSS
# - nss_vector : the vector of NSS computed for a set of values of thresholds
# - best_xss_threshold_tss : the optimum threshold computed with respect to the best TSS 
# - best_xss_threshold_hss : the optimum threshold computed with respect to the best HSS  
# - best_xss_threshold_csi : the optimum threshold computed with respect to the best CSI
# - best_xss_threshold_tss_hss : the optimum threshold computed with respect to the best (TSS+HSS)/2
    n_samples = 100
    step = 1. / n_samples
    
    xss = -1.
    xss_threshold = 0
    Y_best_predicted = numpy.zeros((Y_training.shape))
    tss_vector = numpy.zeros(n_samples)
    hss_vector = numpy.zeros(n_samples)
    csi_vector = numpy.zeros(n_samples)
    xss_threshold_vector = numpy.zeros(n_samples)
    a = probability_prediction.max()
    b = probability_prediction.min()
    for threshold in range(1, n_samples):
        xss_threshold = step * threshold * numpy.abs(a - b) + b
        xss_threshold_vector[threshold] = xss_threshold
        Y_predicted = probability_prediction > xss_threshold
        res = metrics_classification(Y_training > 0, Y_predicted, print_skills=False)
        tss_vector[threshold] = res['tss']
        hss_vector[threshold] = res['hss']
        csi_vector[threshold] = res['csi']
            
    nss_vector = 0.5*((tss_vector/numpy.max(tss_vector)) + (hss_vector/numpy.max(hss_vector)))
    idx_best_nss = numpy.where(nss_vector==numpy.max(nss_vector))  
    print('idx best nss=',idx_best_nss)
    #best NSS
    best_xss_threshold = xss_threshold_vector[idx_best_nss]
    if len(best_xss_threshold)>1:
        best_xss_threshold = best_xss_threshold[0]
        Y_best_predicted = probability_prediction > best_xss_threshold
    else:
        Y_best_predicted = probability_prediction > best_xss_threshold
    print('best NSS')
    metrics_training = metrics_classification(Y_training > 0, Y_best_predicted)
    
    #best TSS
    idx_best_tss = numpy.where(tss_vector==numpy.max(tss_vector))  
    print('idx best tss=',idx_best_tss)
    best_xss_threshold_tss = xss_threshold_vector[idx_best_tss]
    if len(best_xss_threshold_tss)>1:
        best_xss_threshold_tss = best_xss_threshold_tss[0]
        Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
    else:
        Y_best_predicted_tss = probability_prediction > best_xss_threshold_tss
    print('best TSS')
    metrics_training_tss = metrics_classification(Y_training > 0, Y_best_predicted_tss)
    
    #best HSS
    idx_best_hss = numpy.where(hss_vector==numpy.max(hss_vector))  
    print('idx best hss=',idx_best_hss)
    best_xss_threshold_hss = xss_threshold_vector[idx_best_hss]
    if len(best_xss_threshold_hss)>1:
        best_xss_threshold_hss = best_xss_threshold_hss[0]
        Y_best_predicted_hss = probability_prediction > best_xss_threshold_hss
    else:
        Y_best_predicted_hss = probability_prediction > best_xss_threshold_hss
    print('best HSS')
    metrics_training_hss = metrics_classification(Y_training > 0, Y_best_predicted_hss)
    
    #best CSI
    idx_best_csi = numpy.where(csi_vector==numpy.max(csi_vector))  
    print('idx best csi=',idx_best_csi)
    best_xss_threshold_csi = xss_threshold_vector[idx_best_csi]
    if len(best_xss_threshold_csi)>1:
        best_xss_threshold_csi = best_xss_threshold_csi[0]
        Y_best_predicted_csi = probability_prediction > best_xss_threshold_csi
    else:
        Y_best_predicted_csi = probability_prediction > best_xss_threshold_csi
    print('best CSI')
    metrics_training_csi = metrics_classification_weight(Y_training > 0, Y_best_predicted_csi)
    
    
    #best (TSS+HSS)/2
    comb_tss_hss = (hss_vector+tss_vector)/2.
    idx_best_tss_hss = numpy.where(comb_tss_hss==numpy.max(comb_tss_hss)) 
    print('idx best (tss+hss)/2 =',idx_best_tss_hss)
    best_xss_threshold_tss_hss = xss_threshold_vector[idx_best_tss_hss]
    if len(best_xss_threshold_tss_hss)>1:
        best_xss_threshold_tss_hss = best_xss_threshold_tss_hss[0]
        Y_best_predicted_tss_hss = probability_prediction > best_xss_threshold_tss_hss
    else:
        Y_best_predicted_tss_hss = probability_prediction > best_xss_threshold_tss_hss
    print('best (TSS+HSS)/2')
    metrics_training_tss_hss = metrics_classification(Y_training > 0, Y_best_predicted_tss_hss)
    
   

    return best_xss_threshold, metrics_training, nss_vector, best_xss_threshold_tss, best_xss_threshold_hss, best_xss_threshold_csi,best_xss_threshold_tss_hss


#
def metrics_classification(y_real, y_pred, print_skills=True):
# computation of confusion matrix and skill scores (far, pod, acc, hss, tss, fnfp, csi) given in input:
# - y_real :  the actual label vector
# - y_pred :  the vector of predicted probabilities

    cm, far, pod, acc, hss, tss, fnfp, csi = classification_skills(y_real, y_pred)

    if print_skills:
        print ('confusion matrix')
        print (cm)
        print ('false alarm ratio       \t', far)
        print ('probability of detection\t', pod)
        print ('accuracy                \t', acc)
        print ('hss                     \t', hss)
        print ('tss                     \t', tss)
        print ('balance                 \t', fnfp)
        print ('csi                 \t', csi)

    balance_label = float(sum(y_real)) / y_real.shape[0]

    #cm, far, pod, acc, hss, tss, fnfp = classification_skills(y_real, y_pred)

    return {
        "cm": cm,
        "far": far,
        "pod": pod,
        "acc": acc,
        "hss": hss,
        "tss": tss,
        "fnfp": fnfp,
        "balance label": balance_label,
        "csi": csi}
        
        
        
        
def classification_skills(y_real, y_pred):
# computation of confusion matrix and skill scores (far, pod, acc, hss, tss, fnfp, csi) given in input:
# - y_real :  the actual label vector
# - y_pred :  the vector of predicted probabilities


    cm = confusion_matrix(y_real, y_pred)

    if cm.shape[0] == 1 and sum(y_real) == 0:
        a = 0.
        d = float(cm[0, 0])
        b = 0.
        c = 0.
    elif cm.shape[0] == 1 and sum(y_real) == y_real.shape[0]:
        a = float(cm[0, 0])
        d = 0.
        b = 0.
        c = 0.
    elif cm.shape[0] == 2:
        a = float(cm[1, 1])
        d = float(cm[0, 0])
        b = float(cm[0, 1])
        c = float(cm[1, 0])
    TP = a
    TN = d
    FP = b
    FN = c

    if (TP + FP + FN + TN) == 0.:
        if (TP + TN) == 0.:
            acc = 0.  # float('NaN')
        else:
            acc = -100  # float('Inf')
    else:
        acc = (TP + TN) / (TP + FP + FN + TN)

    if TP + FN == 0.:
        if TP == 0.:
            tss_aux1 = 0.  # float('NaN')
        else:
            tss_aux1 = -100  # float('Inf')
    else:
        tss_aux1 = (TP / (TP + FN))

    if (FP + TN) == 0.:
        if FP == 0.:
            tss_aux2 = 0.  # float('NaN')
        else:
            tss_aux2 = -100  # float('Inf')
    else:
        tss_aux2 = (FP / (FP + TN))

    tss = tss_aux1 - tss_aux2

    if ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)) == 0.:
        if (TP * TN - FN * FP) == 0:
            hss = 0.  # float('NaN')
        else:
            hss = -100  # float('Inf')
    else:
        hss = 2 * (TP * TN - FN * FP) / ((TP + FN) *
                                         (FN + TN) + (TP + FP) * (FP + TN))

    if FP == 0.:
        if FN == 0.:
            fnfp = 0.  # float('NaN')
        else:
            fnfp = -100  # float('Inf')
    else:
        fnfp = FN / FP

    if (TP + FN) == 0.:
        if TP == 0.:
            pod = 0  # float('NaN')
        else:
            pod = -100  # float('Inf')
    else:
        pod = TP / (TP + FN)


    if (TP + FP) == 0.:
        if FP == 0.:
            far = 0.  # float('NaN')
        else:
            far = -100  # float('Inf')
    else:
        far = FP / (TP + FP)

    #acc = (a + d) / (a + b + c + d)
    #tpr = a / (a + b)
    #tnr = d / (d + c)
    #wtpr = a / (a + b) * (a + c) / (a + b + c + d) + d / (c + d) * (b + d) / (a + b + c + d)
    #pacc = a / (a + c)
    #nacc = d / (b + d)
    #wacc = a / (a + c) * (a + c) / (a + b + c + d) + d / (b + d) * (b + d) / (a + b + c + d)

    # if the cm has a row or a column equal to 0, we have bad tss
    if TP+FN == 0 or TN+FP == 0 or TP+FP == 0 or TN+FN == 0:
        tss = 0
    if TP+FP+FN==0:
        csi = 0
    else:
        csi = TP/(TP+FP+FN)

 
    return cm.tolist(), far, pod, acc, hss, tss, fnfp, csi








        
