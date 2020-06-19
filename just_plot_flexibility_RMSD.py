'''

Script used to plot the RMSD values in the flexibility analysis

'''



import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib

chains = np.array(['D', 'C', 'B', 'A', 'F', 'E'])

chain_types = np.array(['ZPC\N{SUBSCRIPT MINUS}\N{SUBSCRIPT TWO}', 'ZPN\N{SUBSCRIPT ZERO}',
                        'ZPC\N{SUBSCRIPT MINUS}\N{SUBSCRIPT ONE}', 'ZPN\N{SUBSCRIPT ONE}',
                        'ZPC\N{SUBSCRIPT ZERO}', 'ZPN\N{SUBSCRIPT TWO}'])

print(chain_types)

rmsd_holder = np.load('rmsd_holder_ZPN.npy', allow_pickle=True)
#rmsd_holder = np.load('rmsd_holder_ZPC.npy', allow_pickle=True)


plt.figure(num=None, figsize=(8, 4), dpi=120, facecolor='w', edgecolor='k')
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

for chain_letter in chains:
    rmsd_holder_ = rmsd_holder[36:]

    std = np.std(rmsd_holder_[np.where(rmsd_holder_[:, 1] == chain_letter)][1:, 2].astype(float))

    std_error = stats.sem(rmsd_holder_[np.where(rmsd_holder_[:, 1] == chain_letter)][1:, 2].astype(float))

    plt.errorbar(np.where(chains == chain_letter)[0],
                 np.average(rmsd_holder_[np.where(rmsd_holder_[:, 1] == chain_letter)][1:, 2].astype(float)), yerr=std,
                 fmt='k', capsize=3, elinewidth=1, marker='+')


n = 0
for row in rmsd_holder:
    print(n, row)
    n+=1

for n, element in enumerate(rmsd_holder):


    if element[0] != '0':

        # if 0 <= n <= 36:
        #     plt.scatter(np.where(chains == element[1])[0], element[2].astype(float), c='mediumseagreen')
        #     plt.xticks(range(0, 6), chain_types)

        if 36 <= n <= 59:
            plt.scatter(np.where(chains == element[1])[0], element[2].astype(float), c='coral')
            plt.xticks(range(0, 6), chain_types)

        if 59 <= n <= 90:
            #plt.scatter(np.where(chains == element[1])[0], element[2].astype(float), c='lightsteelblue')
            plt.scatter(np.where(chains == element[1])[0], element[2].astype(float), c='mediumseagreen')
            plt.xticks(range(0, 6), chain_types)

        plt.xlabel('Subdomain')
        plt.ylabel('Subdomain RMSD')



plt.savefig('figure_ZPN_rmsd.svg', dpi=200)
plt.show()