# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 18:59:34 2018

@author: delga
"""
import pandas as pd
import numpy as np
from math import ceil
from  sklearn.neighbors import KDTree 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import copy
def buscar_0inhis(datos,factor = 2):
        histograma = np.histogram(datos)
        if 0 in histograma[0][ceil(histograma[0].__len__()/2):,]:         
            while(True):
                indice =  list(histograma[0]).index(0)
                if indice < ceil(histograma[0].__len__()/2):
                    histograma[0][indice] = 1
                else:
                    break
            valor =  histograma[1][indice]
        else:
            valor = max(histograma[1][:])
        return histograma ,valor

def clear5dnn(d5nn,valor):
    d5NN_mask = d5nn<valor
    d5nn_without_outliers = np.trim_zeros(d5NN_mask*d5nn)
    return d5nn_without_outliers

def fab(d):
    infab = 1/(1+d)
    return infab
    
    
def lower_quartile(data,num):
    ndatos=data.shape[0]
    indice = ceil(num*(ndatos/4))
    return (data[indice][0]+data[indice+1][0])/2

def rhof(sorrounding_region):
    fij = 0
    for d in sorrounding_region:
        fij += fab(d)
    return fij

def rhosmean(knn_index,distance,R,rho):
    rho_s = 0
    cont = 0
    r = []
    for i,d in enumerate(distance):
        if d<=R:
            r.append(rho[knn_index[i]])
            rho_s += rho[knn_index[i]]
            cont += 1
    if cont == 0:
        return 0
    return rho_s/cont #+ np.std(r)/2

def kkddarss(N,X,num):
    k_tree = KDTree(X)
    knn_index = []
    distance = []
    d5NN = []
    Arr = []
    rho = []
    sorrounding_region_Arr = []
    for i in range(N):
        dist, knn_ind = k_tree.query(X[i].reshape(1,-1),k+1)
        knn_index.append(knn_ind[0,1:])
        distance.append(dist[0,1:])
        d5NN.append(np.average(dist[0,1:6]))
    histograma, valor = buscar_0inhis(d5NN,factor = 2)
    d5nn_without_outliers = clear5dnn(d5NN,valor)
    R = max(d5nn_without_outliers)
    dismin = np.min(distance)
    if dismin == 0:
        dismin = 0.01
    for i in range(N):
        sorrounding_region_mask = distance[i]<R
        tem = k
        while np.sum(sorrounding_region_mask)==tem:
            tem = tem + 1
            dist, knn_ind = k_tree.query(X[i].reshape(1,-1),tem+1)
            distance[i] = dist[0,:]
            knn_index[i] = knn_ind[0,:]
            sorrounding_region_mask = distance[i]<R
        if False in sorrounding_region_mask:
            sorrounding_region = np.trim_zeros(sorrounding_region_mask*distance[i])
        else:
            sorrounding_region = distance[i]
        sorrounding_region_Arr.append(sorrounding_region)
        rhov = rhof(sorrounding_region)
        rho.append(rhov)  
        Arr.append((rhov,i))
    dtyp = [('valor',float),('indi',int)]
    Arr = np.array(Arr,dtype=dtyp)
    Arr=np.sort(Arr, order = 'valor')[::-1]
    T = lower_quartile(Arr,num)
    return knn_index, distance, Arr,rho, R,T

if '__main__' == __name__:
    #graficar
    colors_list = list(colors._colors_full_map.values())
#    shuffle(colors_list)
    tmpc=colors_list[0]
    a=colors_list.index('#FF0000')
    colors_list[0]=colors_list[a]
    colors_list[a]=tmpc
    #Aggregation, Jain, Spiral, Flame, Pathbased, R15,'Compound', 'concentric', TCC1 *betha
    for Filename in ['Aggregation', 'Jain', 'Spiral', 'Flame',]:# 'Pathbased', 'R15','Compound', 'concentric', 'D31', 'TCC1']:#,'transactions10k']:
        Dataset = pd.read_csv(Filename+'.csv',  header=None)
        Dataset = np.array(Dataset)
        X = Dataset[:,0:-1]
        Y = Dataset[:,-1]
        N = X.shape[0]
        m = X.shape[1]   
        ####################### ACND ##########################################
        ########################## Normalizo al tamaño#########################
        for i in range(m):
            minimo = np.min(X[:,i])
            X[:,i] = (X[:,i]-minimo)
            maximo = np.max(X[:,i])
            X[:,i] =X[:,i] / maximo
        #######################################################################    
        del(Dataset)
        #########ACND###############
        N = X.shape[0]
        m = X.shape[1]
        k = ceil(max([10,np.log2(N)]))#N*0.03
        print(k)
        #######Aqui iva
        XP = X
        NP = N
        knn_index, distance, Arr,rho, R, T = kkddarss(NP,XP,3) 
        orden_cluster = 0
        resultado = np.zeros(N)
        tag = np.zeros(N,dtype=bool)
        outliers = []
        for index_Arr in range(N):
            if tag[Arr[index_Arr][1]]:
                continue
            if(Arr[index_Arr][0]==0):
                outliers.append(Arr[index_Arr][1])
                tag[Arr[index_Arr][1]] = True
                continue
            if Arr[index_Arr][0] < T:
                continue
            orden_cluster +=1
            tag[Arr[index_Arr][1]]=True
            resultado[Arr[index_Arr][1]] = orden_cluster
            queue = []
            queue.append(Arr[index_Arr][1])
            rhoaf = rhosmean(knn_index[i],distance[i],R,rho)
            while(queue):
                i = queue.pop()
                rhobf = rhosmean(knn_index[i],distance[i],R,rho)
                delta = (abs(np.mean(distance[i])) / (np.mean(distance[i])+(np.std(distance[i]))))
                rhoaf = (rhobf + rhoaf)/2
                localT = rhoaf*delta
                for t in range(k):
                    j = knn_index[i][t]
                    if distance[i][t]>R:            
                        continue
                    if rho[j]>=0 and not(tag[j]):
                        resultado[j] = orden_cluster                                                    
                        if rho[j]>=localT:
                            queue.append(j)
                        else:
                            for jj in range(k):
                                jindex = knn_index[j][jj]
                                if distance[j][jj]<R and not(tag[jindex]):
                                    resultado[jindex] = orden_cluster
                            pass
                    tag[j] = True
            
#        for i in range(N):
#            plt.plot(XP[i][0],XP[i][1],c=colors_list[int(Y[i])*20],marker='o')
#        plt.show()
#            
        for i in range(N):
            plt.plot(XP[i][0],XP[i][1],c=colors_list[int(resultado[i])*30],marker='o')
        plt.axis('off')
        plt.show()
        










