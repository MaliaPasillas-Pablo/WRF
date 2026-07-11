import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def MonthTimeSeries_Wind(Time, Vals, WindSpeed, WindDir, Title, Labels, yaxis=None, Show=False):
    """
    Monthly time series with:
    - Wind direction arrows (top thin panel)
    - Wind speed (taller panel)
    - Vapor pressure
    - Statistics + legend (bottom panel)
    """

    ylabel = yaxis if yaxis is not None else Labels[0]

    ref = np.array(Vals[0])

    mean_diffs = []
    rmsds = []
    ioas = []

    # ===============================
    # Statistics
    # ===============================
    for i in range(1, len(Vals)):
        model = np.array(Vals[i])
        valid = np.isfinite(ref) & np.isfinite(model)
        ref_valid = ref[valid]
        model_valid = model[valid]
        diff = model_valid - ref_valid
        mean_diffs.append(np.mean(diff))
        rmsds.append(np.sqrt(np.mean(diff**2)))
        numerator = np.sum((model_valid - ref_valid)**2)
        denominator = np.sum(
            (np.abs(model_valid - np.mean(ref_valid)) +
             np.abs(ref_valid - np.mean(ref_valid)))**2
        )
        ioa = 1 - numerator / denominator if denominator != 0 else np.nan
        ioas.append(ioa)

    # ===============================
    # Plot
    # ===============================
    if Show:

        fig = plt.figure(figsize=(30,10))

        # Layout: arrows, wind speed, vapor pressure, stats+legend
        gs = fig.add_gridspec(
            4, 1,
            height_ratios=[0.8, 2.5, 4, 1.0],
            hspace=0.05
        )

        ax_dir   = fig.add_subplot(gs[0])
        ax_wspd  = fig.add_subplot(gs[1], sharex=ax_dir)
        ax       = fig.add_subplot(gs[2], sharex=ax_dir)
        ax_stats = fig.add_subplot(gs[3])
        ax_stats.axis('off')  # hide axes

        colors = ['tab:blue','tab:red','tab:orange']

        Time_np = np.array(Time)
        WindSpeed_np = np.array(WindSpeed)
        WindDir_np = np.array(WindDir)

        handles = []
        labels_out = []

        # ===============================
        # WIND DIRECTION ARROWS (TOP BAND)
        # ===============================
        ax_dir.set_yticks([])
        ax_dir.set_xticks([])
        ax_dir.grid(False)
        for spine in ax_dir.spines.values():
            spine.set_visible(False)

        theta = np.deg2rad(270 - WindDir_np)
        U = np.cos(theta)
        V = np.sin(theta)
        step = 2
        ax_dir.set_ylim(-1.2, 1.2)

        ax_dir.quiver(
            Time_np[::step],
            np.zeros(len(Time_np[::step])),
            U[::step],
            V[::step],
            angles='uv',
            scale_units='width',
            scale=60,
            width=0.001  # significantly thinner arrows
        )

        # ===============================
        # WIND SPEED PANEL
        # ===============================
        h_wspd, = ax_wspd.plot(
            Time_np,
            WindSpeed_np,
            '.k',
            markersize=6,
            label="Wind Speed"
        )
        ax_wspd.set_ylabel("Wind Speed (m/s)", fontsize=14)
        ax_wspd.grid(True)
        ax_wspd.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax_wspd.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax_wspd.grid(which='major', axis='x', linestyle='--', alpha=0.5)
        ax_wspd.axhline(y=2, color='red', linewidth=0.8)

        handles.append(h_wspd)
        labels_out.append("Wind Speed")

        # ===============================
        # VAPOR PRESSURE PANEL
        # ===============================
        for i in range(len(Vals)):
            if i == 0:
                h, = ax.plot(Time, Vals[i], '.k', label=Labels[i])
            else:
                h, = ax.plot(
                    Time,
                    Vals[i],
                    linewidth=1.75,  # updated line width
                    color=colors[i-1],
                    label=Labels[i]
                )
            handles.append(h)
            labels_out.append(Labels[i])

        ax.set_ylabel(ylabel, fontsize=18)
        ax.grid(True)
        ax.set_title("")  # no title
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.grid(which='major', axis='x', color='gray', linestyle='--', alpha=0.5)

        # ===============================
        # STATS + LEGEND PANEL
        # ===============================
        top_padding = 0.7  # blank space above stats

        text = ""
        for i in range(len(mean_diffs)):
            text += (
                f"{Labels[i+1]}   "
                f"Bias={mean_diffs[i]:.2f}   "
                f"RMSD={rmsds[i]:.2f}   "
                f"IOA={ioas[i]:.2f}\n"
            )

        ax_stats.text(
            0.02,
            top_padding,
            text,
            fontsize=16,
            bbox=dict(facecolor='white', pad=4),
            verticalalignment='top'
        )

        ax_stats.legend(
            handles,
            labels_out,
            loc='lower right',
            ncol=4,
            fontsize=14,
            frameon=True
        )

        plt.subplots_adjust(bottom=0.12)
        plt.show()

    return [mean_diffs, rmsds, ioas]