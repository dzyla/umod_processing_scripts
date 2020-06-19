import os
import random
import Bio
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
import matplotlib.pyplot as plt
from scipy import ndimage


def load_pdb(path):


    # If using PDB
    # parser = PDBParser(PERMISSIVE=1)

    # if using mmCIF
    parser = MMCIFParser()

    structure = parser.get_structure('structure 1', path)

    return structure


'''
This code is adapted from Evolutron https://github.com/mitmedialab/Evolutron
from GetInterfaces.py

Shared with MIT license.
'''

def save_contacts(structure, chains, out_file):
    # Save only those chains that we are supposed to
    Select = Bio.PDB.Select

    class ConstrSelect(Select):
        def accept_chain(self, chain):
            # print dir(residue)

            if chain.id in chains:
                return 1
            else:
                return 0

    from Bio.PDB import PDBIO
    w = PDBIO()
    w.set_structure(structure)
    randint = random.randint(0, 9999999)
    w.save("TMP" + str(randint) + ".pdb", ConstrSelect())
    # Remove the HETATM and TER lines
    f_tmp = open("TMP" + str(randint) + ".pdb", 'r')
    f_out = open(out_file, 'w')
    for line in f_tmp.readlines():
        if line[0:3] != "TER" and line[0:6] != "HETATM":
            f_out.write(line)
    f_tmp.close()
    f_out.close()
    os.remove("TMP" + str(randint) + ".pdb")



def distance(x, y):
    d = np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2)
    return d


def get_rmsd(matrix1, matrix2):
    diff = (matrix1 - matrix2) ** 2
    sum_ = np.sum(diff)
    div_ = sum_ / diff.shape[0]

    return np.sqrt(div_)

# Here put the path the the mmCIF or PDB file which has the all states from single 3D classification
class3D_states = ['/path/to/multistate/mmcif']


# define the storing lists
state1_matrix = []
state2_matrix = []
rmsd_holder = []
mass_centers = []

for class3d_state in class3D_states:

    umod_class1 = load_pdb(class3d_state)

    # For every state in multiple states
    for state in umod_class1:

        # for all chains in the state
        for chain in state.get_chains():

            state1_matrix = []
            state2_matrix = []

            # this one limits number of chains one would use, put all chains if all should be used
            if chain.id in ['A', 'B', 'C', 'D', 'E', 'F']:

                # get coordinates only for CA atoms
                for residue in umod_class1[0][chain.id]:
                    try:
                        state1_matrix.append(residue['CA'].coord)
                    except:
                        pass

                # for center of mass calculation, not used here
                if chain.id == 'A':
                    reference_matrix = state1_matrix
                    center_of_mass1x = np.sum(np.array(reference_matrix)[:, 0]) / len(reference_matrix)
                    center_of_mass1y = np.sum(np.array(reference_matrix)[:, 1]) / len(reference_matrix)
                    center_of_mass1z = np.sum(np.array(reference_matrix)[:, 2]) / len(reference_matrix)


                for residue in umod_class1[state.id][chain.id]:
                    try:
                        state2_matrix.append(residue['CA'].coord)
                    except:
                        pass


                center_of_mass2x = np.sum(np.array(state2_matrix)[:,0])/len(state2_matrix)
                center_of_mass2y = np.sum(np.array(state2_matrix)[:,1])/len(state2_matrix)
                center_of_mass2z = np.sum(np.array(state2_matrix)[:,2])/len(state2_matrix)

                chain_distance = distance(np.array([center_of_mass1x, center_of_mass1y, center_of_mass1z]),
                                          np.array([center_of_mass2x, center_of_mass2y, center_of_mass2z]))

                center1 = np.array([center_of_mass1x, center_of_mass1y, center_of_mass1z])
                center2 = np.array([center_of_mass2x, center_of_mass2y, center_of_mass2z])

                mass_centers.append([state.id, chain.id, center1, center2])


                # all CA per state
                state1_matrix = np.array(state1_matrix)
                state2_matrix = np.array(state2_matrix)

                #calculate RMSD between states
                rmsd = get_rmsd(state1_matrix, state2_matrix)

                # store the result
                rmsd_holder.append([state.id, chain.id, rmsd])



# the order of chain in the structure
chains = np.array(['D', 'C', 'B', 'A', 'F', 'E'])


rmsd_holder = np.stack(rmsd_holder)
data = rmsd_holder[:, 2].astype(float)

# save data for plotting
np.save('rmsd_holder_ZPN.npy', rmsd_holder)
np.save('mass_centers_ZPN_distance.npy', mass_centers)


# Plot errors
for chain_letter in chains:

    std = np.std(rmsd_holder[np.where(rmsd_holder[:, 1] == chain_letter)][1:, 2].astype(float))

    plt.errorbar(np.where(chains == chain_letter)[0],
                 np.average(rmsd_holder[np.where(rmsd_holder[:, 1] == chain_letter)][1:, 2].astype(float)), yerr=std,
                 fmt='.k', capsize=3)


# plot all RMSDs
for element in rmsd_holder:

    if element[0] != '0':
        plt.scatter(np.where(chains == element[1])[0], element[2].astype(float), c='tab:blue')
        plt.xlabel('Fitted Chains')
        plt.xticks(range(0, 6), chains)
        plt.ylabel('Chain RMSD')


plt.show()