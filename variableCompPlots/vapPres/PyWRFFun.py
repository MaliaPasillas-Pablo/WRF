import numpy as np
import matplotlib.pyplot as plt

def MonthTimeSeries(Time, Vals, Title, Labels, yaxis=None, Show=False):
    """
    Plot monthly time series and compute stats.

    Parameters
    ----------
    Time : sequence-like
        Time axis.
    Vals : list of array-like
        First element is the reference series; the rest are model series (1–4).
    Title : str
        Figure title.
    Labels : list of str
        Labels for each series (same length as Vals).
    yaxis : str or None, optional
        Custom y-axis label. If None, defaults to Labels[0].
    Show : bool, optional
        If True, make the plots.

    Returns
    -------
    [mean_diffs, rmsds, ioas] : list of lists
        Statistics for each model series vs. the reference.
    """
    # === GAP SETTINGS (adjust these) ===
    gap_top_to_stats = 0.17    # vertical spacing between top plot and stats box
    gap_stats_to_bottom = 1e-15  # vertical spacing between stats and bottom plot
    gap_bottom_to_legend = 0.15  # vertical spacing between bottom plot and legend

    # === Plot height ratios ===
    height_top = 1
    height_stats = 0.1
    height_bottom = 1
    height_legend = 0.2
    # ============================

    n_vals = len(Vals)
    assert 1 <= n_vals <= 5, "Vals must contain between 1 and 5 elements."

    ylabel = yaxis if yaxis is not None else Labels[0]

    ref = np.array(Vals[0])
    mean_diffs, rmsds, ioas = [], [], []

    # === Compute stats for each model ===
    for i in range(1, n_vals):
        model = np.array(Vals[i])
        valid_mask = ref != -9999
        ref_valid = ref[valid_mask]
        model_valid = model[valid_mask]

        if len(ref_valid) == 0:
            mean_diffs.append(np.nan)
            rmsds.append(np.nan)
            ioas.append(np.nan)
            continue

        diff = model_valid - ref_valid
        mean_diffs.append(np.mean(diff))
        rmsds.append(np.sqrt(np.mean(diff ** 2)))

        numerator = np.sum((model_valid - ref_valid) ** 2)
        denominator = np.sum((np.abs(model_valid - np.mean(ref_valid)) +
                              np.abs(ref_valid - np.mean(ref_valid))) ** 2)
        ioa = 1 - numerator / denominator if denominator != 0 else np.nan
        ioas.append(ioa)

    # === Plot only after all stats are computed ===
    if Show:
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[1, 0.11, 1, 0.2])

        ax_top = fig.add_subplot(gs[0])
        ax_stats = fig.add_subplot(gs[1])
        ax_bot = fig.add_subplot(gs[2])
        ax_leg = fig.add_subplot(gs[3])

        ax_stats.axis('off')
        ax_leg.axis('off')

        colors = ['tab:blue', 'tab:red', 'tab:blue', 'tab:orange']

        # === Top Plot ===
        t_top = np.array(Time[0:359])
        v0_top = np.array(Vals[0][0:359])
        mask_top = v0_top != -9999
        ax_top.plot(t_top[mask_top], v0_top[mask_top], '.k', label=Labels[0])
        for j in range(1, n_vals):
            ax_top.plot(Time[0:359], Vals[j][0:359], linewidth=1+(4-j), label=Labels[j], color=colors[j-1])
        ax_top.set_ylabel(ylabel, fontsize=22)
        ax_top.set_title(Title, fontsize=25)
        ax_top.tick_params(labelsize=18)
        ax_top.grid(True)

        # === Bottom Plot ===
        t_bot = np.array(Time[359:718])
        v0_bot = np.array(Vals[0][359:718])
        mask_bot = v0_bot != -9999
        ax_bot.plot(t_bot[mask_bot], v0_bot[mask_bot], '.k', label=Labels[0])
        for j in range(1, n_vals):
            ax_bot.plot(Time[359:718], Vals[j][359:718], linewidth=1+(4-j), label=Labels[j], color=colors[j-1])
        ax_bot.set_ylabel(ylabel, fontsize=18)
        ax_bot.set_xlabel("Time", fontsize=18)
        ax_bot.tick_params(labelsize=14)
        ax_bot.grid(True)

        # === Stats Boxes ===
        mean_text = "\n".join([f"MeanBias: {Labels[j]}: {mean_diffs[j-1]:.2f}" for j in range(1, n_vals)])
        rmsd_text = "\n".join([f"RMSD:     {Labels[j]}: {rmsds[j-1]:.2f}" for j in range(1, n_vals)])
        ioa_text = "\n".join([f"IOA:      {Labels[j]}: {ioas[j-1]:.2f}" for j in range(1, n_vals)])

        ax_stats.text(0.01, 1.0, mean_text, 
                      transform=ax_stats.transAxes, 
                      fontsize=20, 
                      va='top',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
        ax_stats.text(0.35, 1.0, rmsd_text, 
                      transform=ax_stats.transAxes,
                      fontsize=20, 
                      va='top',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
        ax_stats.text(0.68, 1.0, ioa_text, 
                      transform=ax_stats.transAxes,
                      fontsize=20, 
                      va='top',
                      bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

        # === Legend Box ===
        handles, labels = ax_top.get_legend_handles_labels()
        ax_leg.legend(handles, labels, loc='center', fontsize=16, frameon=True, ncol=1)

        # === Layout ===
        plt.tight_layout()
        plt.subplots_adjust(hspace=gap_top_to_stats)

        # Adjust bottom plot position
        fig.canvas.draw()
        pos_stats = ax_stats.get_position()
        pos_bot = ax_bot.get_position()
        current_gap = pos_stats.y0 - pos_bot.y1

        if current_gap > gap_stats_to_bottom:
            shift = current_gap - gap_stats_to_bottom
            ax_bot.set_position([pos_bot.x0,
                                 pos_bot.y0 + shift,
                                 pos_bot.width,
                                 pos_bot.height])

        # Adjust legend position
        pos_bot_new = ax_bot.get_position()
        pos_leg = ax_leg.get_position()
        current_gap_bottom = pos_leg.y1 - pos_bot_new.y0

        if current_gap_bottom > gap_bottom_to_legend:
            shift_leg = current_gap_bottom - gap_bottom_to_legend
            ax_leg.set_position([pos_leg.x0,
                                 pos_leg.y0 - shift_leg,
                                 pos_leg.width,
                                 pos_leg.height])

        plt.show()

    return [mean_diffs, rmsds, ioas]


