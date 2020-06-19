import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data from cryosparc
fsc_data = pd.read_csv('P13_J357_fsc_iteration_020.txt', delimiter='\t')


# Split data from PHENIX
fsc_data_model_unmasked = pd.read_csv('2chains_cryosparc.txt', delimiter=' ')
fsc_data_model_masked = pd.read_csv('cryosparc_model_masked.txt', delimiter=' ')

# Map information
pixel_size = 1.084
box_size = 320

# output plot name
out_name = 'FSC_map_model.svg'

number_of_ticks = 10

plt.plot(fsc_data['wave_number'], fsc_data['fsc_tightmask'], label='Tight')
plt.plot(fsc_data['wave_number'], fsc_data['fsc_noisesub'], label='Corrected')
#plt.plot(fsc_data_model_unmasked['d_inv']*320*1.084, fsc_data_model_unmasked['fsc'], label='Unmasked model')
plt.plot(fsc_data_model_masked['d_inv']*box_size*pixel_size, fsc_data_model_masked['fsc'], label='Masked model')
plt.locator_params(numticks=number_of_ticks)
plt.xticks(np.linspace(0,160,number_of_ticks), np.around(pixel_size*box_size/np.linspace(0,160,number_of_ticks),decimals=2))

plt.hlines(0.143, 0, 160, linestyles='--')
plt.hlines(0.5, 0, 160, linestyles='--')

plt.xlim(15, 155)

plt.xlabel('Resolution (Å)')
plt.ylabel('FSC')
plt.legend()

plt.savefig(out_name)

plt.show()