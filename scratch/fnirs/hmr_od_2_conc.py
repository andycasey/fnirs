
import numpy as np
from .get_extinctions import get_extinctions

def hmr_od_2_conc(dod, sd, ppf):
    """
    Convert OD to concentrations.

    Convert intensity data to concentration.

    Args:
        dod (np.ndarray): The change in OD (#time points x #channels).
        sd (dict): The SD structure.
        ppf (list): Partial pathlength factors for each wavelength.

    Returns:
        np.ndarray: The concentration data (#time points x 3 x #SD pairs).
    """

    assert dod.shape[0] > dod.shape[1], "I don't believe that you have more channels than time points"

    n_wav = len(sd['Lambda'][0])
    ml = sd['MeasList']

    if len(ppf) != n_wav:
        raise ValueError('The length of PPF must match the number of wavelengths in SD.Lambda')

    n_t_pts = dod.shape[0]

    e = get_extinctions(sd['Lambda'][0])
    if 'SpatialUnit' not in sd or sd['SpatialUnit'][0] == 'mm':
        e = e[:, :2] / 10  # convert from /cm to /mm
    elif sd['SpatialUnit'][0] == 'cm':
        e = e[:, :2] # hbo, hbr

    einv = np.linalg.solve(e.T @ e, np.eye(e.shape[1])) @ e.T
    

    lst = np.where(ml[:, 3] == 1)[0]
    dc = np.zeros((n_t_pts, 3, len(lst)))
    rhos = np.zeros(len(lst))
    for idx, idx1 in enumerate(lst):
        idx2 = np.where((ml[:, 3] > 1) & (ml[:, 0] == ml[idx1, 0]) & (ml[:, 1] == ml[idx1, 1]))[0]
        rho = np.linalg.norm(sd['SrcPos'][ml[idx1, 0] - 1, :] - sd['DetPos'][ml[idx1, 1] - 1, :])
        dc[:, :2, idx] = (einv @ (dod[:, np.concatenate(([idx1], idx2))] / (np.ones((n_t_pts, 1)) * rho * np.array(ppf))).T).T
        rhos[idx] = rho

    # dc index 1 is hbo, hbr, hbt
    # hbt = hbo + hbr
    dc[:, 2, :] = dc[:, 0, :] + dc[:, 1, :]
    return dc #, rhos
