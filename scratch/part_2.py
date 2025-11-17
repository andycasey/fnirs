import numpy as np
from fnirs.physiology_regression_glm import physiology_regression_glm
from fnirs.perform_pca_regression import perform_pca_regression
from fnirs.plot_correlation_matrices import plot_correlation_matrices

# Load preprocessed data from Part 1.
data = np.load('Data_for_Part_II.npz', allow_pickle=True)
dc = data['dc']
sd = data['sd'].item()
bad_channels = data['bad_channels']
ss_list = data['ss_list']
phys_data = data['phys_data']

# Solution 1: Short-Channel Regression
additional_regressors = np.array([])
ss_list = [8, 29, 52, 66, 75, 92, 112, 125]

#dc_only_sc, _ = physiology_regression_glm(dc, sd, ss_list, additional_regressors)


# Solution 2: Independent Physiological Measurements
additional_regressors = phys_data
dc_only_phys, _ = physiology_regression_glm(dc, sd, ss_list, additional_regressors)


raise a

# Solution 3: Short Channel Regression and Systemic Physiology
additional_regressors = phys_data
dc_sc_phys, stats = physiology_regression_glm(dc, sd, ss_list, additional_regressors)

# Solution 4: Employ PCA to remove the 1st component
n_sv = [1, 1]
dc_pca_one = perform_pca_regression(dc, sd, n_sv, bad_channels)

# Plot correlation matrices of each case
plot_correlation_matrices(dc, dc_only_sc, dc_only_phys, dc_sc_phys, dc_pca_one, sd, bad_channels)