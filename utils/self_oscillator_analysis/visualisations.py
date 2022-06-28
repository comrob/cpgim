from matplotlib.pyplot import Figure
import matplotlib.cm

import numpy as np
COLORS = [
    "orange", "lime", "royalblue", "crimson", "aquamarine", "gold", "teal", "fuchsia", "peru", "olive"]


def compare_syncing(records, fig: Figure, labels, osc_phase_label,
                    oscillator_phase_name="phs", perturbation_signal_name="pert"):
    t = records[0]["t"]
    base_id = labels.index(osc_phase_label)

    ep_phases = [record[oscillator_phase_name][:, 0] for record in records]
    perturbations = [record[perturbation_signal_name] for record in records]

    pert_sels = [np.where(perturbation > 0.9) for perturbation in perturbations]
    cmap = matplotlib.cm.get_cmap('Spectral')
    f = fig.subplots(2, 1)
    figr_phs = f[0]
    figr_evol = f[1]

    for i, lab in enumerate(labels):
        col_val = i/(len(labels)-1)

        figr_phs.plot(t, ep_phases[i] - ep_phases[base_id], label="{}".format(lab), color=cmap(col_val))
        x = np.mod(ep_phases[i][:, 0], 2 * np.pi/64)
        dif_x = np.max(x) - np.min(x)
        figr_evol.plot(t, x + dif_x * i, alpha=0.3, label="{}".format(lab), color=cmap(col_val))
        figr_evol.plot(t[pert_sels[i]], x[pert_sels[i]] + dif_x * i, '.', color=cmap(col_val))

    figr_phs.legend()
    figr_evol.legend()


def observed_freqs(records, fig: Figure, frequencies, oscillator_phase_name="phs", obs_interval=(10, -1)):
    t = records[0]["t"]
    start, end = obs_interval

    phases = [record[oscillator_phase_name][:, 0, 0] for record in records]
    obs_vels = [(phase[end] - phase[start])/(t[end] - t[start]) for phase in phases]
    beat_vels =  np.asarray(frequencies)/np.asarray(obs_vels)

    cmap = matplotlib.cm.get_cmap('Spectral')

    f = fig.subplots(1, 1)
    figr_obs = f

    figr_obs.plot(np.asarray(frequencies), beat_vels)

    for i, lab in enumerate(frequencies):
        col_val = i/(len(frequencies)-1)

        figr_obs.plot([frequencies[i]], [beat_vels[i]], 'x', color=cmap(col_val))

