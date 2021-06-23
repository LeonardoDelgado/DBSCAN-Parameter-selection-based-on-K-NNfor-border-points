import numpy as np
import os
from math import ceil, floor
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter1d
import copy

def DBSC_L(point_cloud_,samples = 4, resolution  = 10):

    point_cloud = point_cloud_ * resolution
    temp = point_cloud.astype('int')
    temp_df = pd.DataFrame()

    for i in range(temp.shape[1]):
        temp_df[i.__str__()] = temp[:,i]
    temp_df_ = pd.DataFrame()
    for i in range(point_cloud_.shape[1]):
        temp_df_[i.__str__()] = point_cloud_[:,i]


    point_cloud = np.unique(temp, axis=0)
    temp_df_unique =  pd.DataFrame()
    for i in range(point_cloud.shape[1]):
        temp_df_unique[i.__str__()] = point_cloud[:,i]

    columns = list(set(temp_df.columns))
    temp_df = temp_df.reset_index()
    temp_df_ = temp_df_.reset_index()


    tree_pc = KDTree(point_cloud)
    dist_t = []
    count_t = []

    for i in range(point_cloud.shape[0]):
        dist, ind = tree_pc.query(point_cloud[i, :].reshape(1, -1), k=samples + 1)
        dist_t.append(np.mean(dist[0, 1:]))
    dist = ceil(np.mean(dist_t))

    for i in range(point_cloud.shape[0]):
        count = tree_pc.query_radius(point_cloud[i, :].reshape(1, -1), dist, count_only=True)
        count_t.append(count[0])

    n_samples = floor(np.mean(count_t)) -2#- 1

    clustering = DBSCAN(eps=dist, min_samples=n_samples).fit(point_cloud)
    temp_df_unique['labels'] = clustering.labels_
    labels = clustering.labels_
    labels[clustering.core_sample_indices_] = -1
    labels = labels != -1
    temp_df_unique['border'] = labels



    temp_df = temp_df.merge(temp_df_unique,on=columns)

    temp_df_ = temp_df_.merge(temp_df,on='index')


    return clustering, point_cloud, temp_df_.drop('index',axis=1)





def DBSC_O(point_cloud, save = False):
    k = 2*2-1
    # point_cloud = point_cloud * resolution
    # temp = point_cloud.astype('int')
    temp_df = pd.DataFrame()

    for i in range(point_cloud.shape[1]):
        temp_df[i.__str__()] = point_cloud[:,i]

    # point_cloud = np.unique(temp, axis=0)
    # temp_df_unique =  pd.DataFrame()
    # for i in range(point_cloud.shape[1]):
    #     temp_df_unique[i.__str__()] = point_cloud[:,i]



    tree_pc = KDTree(point_cloud)
    dist_t = []
    count_t = []

    for i in range(point_cloud.shape[0]):
        dist, ind = tree_pc.query(point_cloud[i, :].reshape(1, -1), k=k+1)
        dist_t.append(np.max(dist[0, 1:]))
    dist_t.sort(reverse = True)
    par = np.array([[i,dist_t[i]] for i in range(len(dist_t))])
    plt.plot(par[:,1])

    # smooth
    smooth = gaussian_filter1d(np.array(dist_t), 20)

    # compute second derivative
    smooth_d2 = np.gradient(np.gradient(smooth))

    # find switching points
    dist = np.where(np.diff(np.sign(smooth_d2)))[0]

    plt.plot(smooth, label='Smoothed Data')
    plt.plot(smooth_d2 / np.max(smooth_d2), label='Second Derivative (scaled)')

    for i, infl in enumerate(dist, 1):
        plt.axvline(x=infl, color='k', label=f'Inflection Point {i}')
        if i > 3:
            break
    plt.legend(bbox_to_anchor=(1.55, 1.0))
    if save != False:
        plt.savefig(fname=save + '.pdf', bbox_inches = 'tight')
    plt.cla()
    plt.clf()
    # plt.show()

    dist = dist_t[dist[0]]
    print(dist)



    # n_samples = 4
    n_samples = k+1

    clustering = DBSCAN(eps=dist, min_samples=n_samples).fit(point_cloud)
    temp_df['labels'] = clustering.labels_
    labels = clustering.labels_
    labels[clustering.core_sample_indices_] = -1
    labels = labels != -1
    temp_df['border'] = labels
    return clustering, point_cloud, temp_df,dist


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    import os
    try:
        os.mkdir('Data/Plots')
    except:
        print('Plot folder already exits')
    colors_list = list(colors._colors_full_map.values())
    #    shuffle(colors_list)
    tmpc = colors_list[0]
    a = colors_list.index('#FF0000')
    colors_list[0] = colors_list[a]
    colors_list[a] = tmpc
    samples = 30
    #filename = 'Jain'
    for filename in ['Pathbased', 'R15','Compound', 'concentric', 'D31',  'Jain', 'Spiral','Flame']:#['Pathbased', 'R15','Compound', 'concentric', 'D31', 'TCC1', 'Jain', 'Spiral','Flame' ]:#, 'Jain', 'Spiral',
                     #'Flame' ]:  # 'Pathbased', 'R15','Compound', 'concentric', 'D31', 'TCC1']:#,'transactions10k']:
        Dataset = pd.read_csv('Data/'+ filename + '.csv', header=None)
        X = Dataset.iloc[:, 0:-1].values
        y = Dataset.iloc[:, -1].values
        #point_cloud = X
        #clustering, X, temp_df,eps = DBSC_O(X, save = 'Data/Plots/' + filename + '_original_debscan_threshold_' + str(samples))
        clustering, X, temp_df = DBSC_L(X, samples=samples, resolution=100)

        X = temp_df.iloc[:, 0:-2].values
        #y = temp_df.iloc[:, -2:-1].values

        N = X.shape[0]
        resultado  = temp_df.iloc[:,-2]#".astype('int')
        for i in range(N):
            plt.plot(X[i][0], X[i][1], c=colors_list[int(resultado[i]) * 15], marker='o')
        plt.axis('off')
        #plt.savefig(fname = 'Data/Plots/' + filename + 'original_debscan_threshold_'+ eps.__str__() + '_' +'.pdf')
        plt.savefig('Data/Plots/'+filename + 'proposal_debscan_samples_' + str(samples) +'.pdf')
        plt.show()


        for i in range(N):
            plt.plot(X[i][0], X[i][1], c=colors_list[int(y[i]) * 15], marker='o')
        plt.axis('off')
        #plt.savefig(fname = 'Data/Plots/' + filename + 'original_debscan_threshold_'+ eps.__str__() + '_' +'.pdf')
        plt.savefig('Data/Plots/'+filename +'.pdf')
        plt.show()





