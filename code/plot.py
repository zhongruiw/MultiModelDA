import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ─────────────────────────────────────────────────────────────────────────────
# Global style
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':       'DejaVu Sans',

    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   10,
    'figure.titlesize':  14,

    'axes.linewidth':    0.8,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
})


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_corr_rmse_box(ax, truth_arr, mean_arr, warmup, color='black'):
    """Annotate a single-method Corr / RMSE box (top-right corner)."""
    truth_i = truth_arr[warmup:]
    mean_i = mean_arr[warmup:]
    corr = np.corrcoef(truth_i, mean_i)[0, 1]
    rmse = np.sqrt(np.mean((truth_i - mean_i) ** 2))
    ax.text(
        0.99, 0.94,
        f'Corr = {corr:.3f}\nRMSE = {rmse:.3f}',
        transform=ax.transAxes, fontsize=9, fontweight='bold',
        va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.6),
    )


def _add_corr_rmse_box_multi(ax, truth_arr, mean_list, colors, warmup,
                             box_x=0.995, box_y=0.95,
                             spacing_x=0.05, spacing_y=0.12, pad=0.00):
    """Annotate a multi-method Corr / RMSE box with colour-coded values."""
    from matplotlib.patches import FancyBboxPatch

    corrs, rmses = [], []
    for m in mean_list:
        t, p = truth_arr[warmup:], m[warmup:]
        corrs.append(f'{np.corrcoef(t, p)[0, 1]:.3f}')
        rmses.append(f'{np.sqrt(np.mean((t - p) ** 2)):.3f}')
    n = len(corrs)

    # --- measure the widest value string in axes coords -----------
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    all_strings = corrs + rmses
    max_width = 0
    for s in all_strings:
        txt = ax.text(0, 0, s, transform=ax.transAxes,
                      fontsize=9, fontweight='bold')
        bb = txt.get_window_extent(renderer=renderer)
        bb_ax = bb.transformed(ax.transAxes.inverted())
        max_width = max(max_width, bb_ax.width)
        txt.remove()
    spacing_x = max_width + pad

    values_right = box_x
    label_x = values_right - n * spacing_x

    # --- measure label width for background box -------------------
    label_width = 0
    for label in ['Corr:  ', 'RMSE:']:
        txt = ax.text(0, 0, label, transform=ax.transAxes,
                      fontsize=9, fontweight='bold')
        bb = txt.get_window_extent(renderer=renderer)
        bb_ax = bb.transformed(ax.transAxes.inverted())
        label_width = max(label_width, bb_ax.width)
        txt.remove()

    # --- measure row height ---------------------------------------
    txt = ax.text(0, 0, '0.000', transform=ax.transAxes,
                  fontsize=9, fontweight='bold')
    bb = txt.get_window_extent(renderer=renderer)
    bb_ax = bb.transformed(ax.transAxes.inverted())
    row_height = bb_ax.height
    txt.remove()

    # --- draw semi-transparent background box ---------------------
    box_pad = -0.005
    bbox_left = label_x - label_width - box_pad
    bbox_right = values_right + box_pad
    bbox_top = box_y + box_pad
    bbox_bottom = box_y - spacing_y - row_height - box_pad

    bg = FancyBboxPatch(
        (bbox_left, bbox_bottom),
        bbox_right - bbox_left,
        bbox_top - bbox_bottom,
        boxstyle="round,pad=0.008",
        facecolor='white', edgecolor='gray',
        alpha=0.7, linewidth=0.5,
        transform=ax.transAxes, zorder=4,
    )
    ax.add_patch(bg)

    # --- row labels -----------------------------------------------
    text_zorder = 5
    for label, dy in [('Corr:  ', 0), ('RMSE:', spacing_y)]:
        ax.text(
            label_x, box_y - dy, label,
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='right', color='black', zorder=text_zorder,
        )

    # --- colour-coded values --------------------------------------
    for j, c in enumerate(colors[:n]):
        x_pos = values_right - (n - 1 - j) * spacing_x
        ax.text(x_pos, box_y, corrs[j],
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                va='top', ha='right', color=c, zorder=text_zorder)
        ax.text(x_pos, box_y - spacing_y, rmses[j],
                transform=ax.transAxes, fontsize=9, fontweight='bold',
                va='top', ha='right', color=c, zorder=text_zorder)


def _regime_panel(ax, time, S=None, prior_weights=None, posterior_weights=None,
                  line_width=1.5):
    """Draw a regime / weight panel and return (lines, labels) for legend."""
    lines, labels = [], []
    if S is not None:
        l, = ax.plot(time, S, 'k', linewidth=line_width)
        lines.append(l); labels.append('True Regime')
    if prior_weights is not None:
        l, = ax.plot(time, prior_weights, 'b--', linewidth=line_width)
        lines.append(l); labels.append('Prior Weight')
    if posterior_weights is not None:
        l, = ax.plot(time, posterior_weights, 'r--', linewidth=line_width)
        lines.append(l); labels.append('Posterior Weight')
    ax.set_ylim([-0.1, 1.1])
    ax.set_title('Regime')
    return lines, labels


def format_param(value):
    """Format a parameter value, using fractions for non-integer rationals."""
    from fractions import Fraction
    frac = Fraction(value).limit_denominator(100)
    return str(frac.numerator) if frac.denominator == 1 else f'{frac.numerator}/{frac.denominator}'


# ─────────────────────────────────────────────────────────────────────────────
# Model diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def plot_L63_regimes(t, z_truth, S, regimes, dt,
                     z_indep_list=None, t_display_max=200, max_lag_time=20.0,
                     figsize=(14, 11), savefig=None):
    """Lorenz-63 regime-switching diagnostics."""
    from matplotlib import gridspec
    from scipy.stats import gaussian_kde
    from statsmodels.tsa.stattools import acf

    n_regimes = len(regimes)
    max_lag = int(max_lag_time / dt)
    idx_display = t <= t_display_max
    lag_axis = np.arange(max_lag + 1) * dt
    regime_labels = [f'{k}' for k in range(n_regimes)]

    # Build panel data
    panel_data = []
    for k in range(n_regimes):
        r = regimes[k]
        title = (f"Regime {k} ")
        z_k = z_indep_list[k]
        panel_data.append((z_k, z_k, acf(z_k, nlags=max_lag, fft=True), title))
    panel_data.append((z_truth, z_truth,
                       acf(z_truth, nlags=max_lag, fft=True), 'Two-regime model'))

    n_ts_rows = len(panel_data)
    n_rows = n_ts_rows + 1

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        n_rows, 3,
        width_ratios=[5, .8, .8],
        height_ratios=[1] * n_ts_rows + [1],
        hspace=0.3, wspace=0.15,
        left=0.07, right=0.97, top=0.95, bottom=0.06,
    )

    ax_ts_first = None
    for row, (z_ts, z_vals, acf_vals, title) in enumerate(panel_data):
        # Time series
        share = dict(sharex=ax_ts_first) if ax_ts_first else {}
        ax_ts = fig.add_subplot(gs[row, 0], **share)
        if ax_ts_first is None:
            ax_ts_first = ax_ts
        ax_ts.plot(t[idx_display], z_ts[idx_display], color='k', lw=0.8, rasterized=True)
        ax_ts.set_ylabel('$z$')
        ax_ts.set_title(f'{title}')
        if row < n_ts_rows:
            plt.setp(ax_ts.get_xticklabels(), visible=False)

        # PDF
        ax_pdf = fig.add_subplot(gs[row, 1])
        valid = z_vals[~np.isnan(z_vals)]
        kde = gaussian_kde(valid)
        z_grid = np.linspace(valid.min(), valid.max(), 300)
        ax_pdf.plot(kde.evaluate(z_grid), z_grid, color='k', lw=0.8)
        ax_pdf.set_ylim(ax_ts.get_ylim())
        ax_pdf.set_yticklabels([])
        if row == 0:
            ax_pdf.set_title('PDF')
        if row == n_ts_rows - 1:
            ax_pdf.set_xlabel('density')

        # ACF
        ax_acf = fig.add_subplot(gs[row, 2])
        ax_acf.plot(lag_axis, acf_vals, color='k', lw=0.8)
        ax_acf.set_xlim(0, max_lag_time)
        if row == 0:
            ax_acf.set_title('ACF')
        if row == n_ts_rows - 1:
            ax_acf.set_xlabel('lag')

    # Regime sequence
    ax_regime = fig.add_subplot(gs[n_ts_rows, 0], sharex=ax_ts_first)
    ax_regime.plot(t[idx_display], S[idx_display], color='k', lw=0.8, drawstyle='steps-post')
    ax_regime.set_ylim(-0.15, n_regimes - 1 + 0.15)
    ax_regime.set_yticks(range(n_regimes))
    ax_regime.set_yticklabels(regime_labels)
    ax_regime.set_xlabel('$t$')
    ax_regime.set_ylabel('regime')
    ax_regime.set_title('Regime sequence')
    ax_ts_first.set_xlim(0, t_display_max)

    return fig


def plot_L63_trajectories(x_truth, y_truth, z_truth, dt,
                          sel0=10000, sel1=20000, interv=10, figsize=(10, 6)):
    """Sample 1-D, 2-D, and 3-D trajectories of the Lorenz-63 system."""
    xaxis = np.arange(sel0 * dt, sel1 * dt, interv * dt)

    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    gs0 = GridSpec(1, 2, figure=fig)

    # Left column – individual trajectories
    gs00 = gs0[0].subgridspec(3, 6)
    for idx, (arr, var) in enumerate([(x_truth, 'x'), (y_truth, 'y'), (z_truth, 'z')]):
        ax = fig.add_subplot(gs00[idx, :])
        ax.plot(xaxis, arr[sel0:sel1:interv])
        ax.set_xlim(xaxis[0], xaxis[-1])
        ax.set_title(f'({chr(97 + idx)}) Sample trajectory of {var}')
    ax.set_xlabel('$t$')

    # Right column – phase portraits
    gs01 = gs0[1].subgridspec(2, 7)
    pairs = [
        (gs01[0, :3], x_truth, y_truth, 'x', 'y', 'd'),
        (gs01[0, 4:], y_truth, z_truth, 'y', 'z', 'e'),
        (gs01[1, :3], z_truth, x_truth, 'z', 'x', 'f'),
    ]
    for spec, a, b, xl, yl, lbl in pairs:
        ax = fig.add_subplot(spec)
        ax.plot(a, b, lw=0.5)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f'({lbl}) 2D trajectory of {xl} and {yl}')

    ax7 = fig.add_subplot(gs01[1, 4:], projection='3d')
    ax7.plot(x_truth, y_truth, z_truth, lw=0.5)
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_zlabel('z')
    ax7.set_title('(g) 3D trajectory of x, y, and z')
    ax7.grid(False)
    plt.tight_layout()


def plot_ou_series_pdf_acf(dt, sel0, sel1, interv, xlim,
                           ur_list, ui_list, v_list, labels, colors,
                           max_lag=4000, figsize=(8, 3.5)):
    """Time series, PDFs, and ACFs for complex OU variables (u_R, u_I, v)."""
    from scipy.stats import gaussian_kde, norm
    from statsmodels.tsa.stattools import acf
    xaxis = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    lag_axis = dt * np.arange(max_lag + 1)
    fig = plt.figure(figsize=figsize)
    spec = fig.add_gridspec(nrows=3, ncols=3,
                            width_ratios=[5, 0.8, 0.8], height_ratios=[1, 1, 1])
    var_names = [r'$u_R$', r'$u_I$', r'$v$']
    data_groups = [ur_list, ui_list, v_list]
    legend_lines, legend_labels = [], []

    ax_ts0, ax_pdf0, ax_acf0 = None, None, None
    for row, (var_list, var_label) in enumerate(zip(data_groups, var_names)):
        ax_ts = fig.add_subplot(spec[row, 0], sharex=ax_ts0)
        ax_pdf = fig.add_subplot(spec[row, 1], sharex=ax_pdf0)
        ax_acf = fig.add_subplot(spec[row, 2], sharex=ax_acf0)
        if row == 0:
            ax_ts0, ax_pdf0, ax_acf0 = ax_ts, ax_pdf, ax_acf

        for i, data in enumerate(var_list):
            samples = data[sel0:sel1]
            series = data[sel0:sel1:interv]
            line, = ax_ts.plot(xaxis, series, color=colors[i], label=labels[i], linewidth=1.)
            if row == 0 and i == 0:
                legend_lines.append(line)
                legend_labels.append(labels[i])
            kde = gaussian_kde(samples)
            x_pdf = np.linspace(samples.min(), samples.max(), 300)
            ax_pdf.plot(kde.evaluate(x_pdf), x_pdf, color=colors[i], linewidth=1.)
            ax_acf.plot(lag_axis, acf(samples, nlags=max_lag, fft=True), color=colors[i], linewidth=1.)

        ax_ts.set_xlim(xlim)
        ax_ts.set_ylabel(var_label)
        ax_acf.set_xlim(0, lag_axis[-1])

        if row < 2:
            plt.setp(ax_ts.get_xticklabels(), visible=False)
            plt.setp(ax_pdf.get_xticklabels(), visible=False)
            plt.setp(ax_acf.get_xticklabels(), visible=False)
        else:
            ax_ts.set_xlabel('$t$')
            ax_pdf.set_xlabel('density')
            ax_acf.set_xlabel('lag')

        if row == 0:
            ax_ts.set_title('Time Series')
            ax_pdf.set_title('PDF')
            ax_acf.set_title('ACF')

    plt.tight_layout(rect=[0, 0, 1, 0.97])


def plot_topobaro_series(dt, sel0, sel1, interv, xlim,
                     data_groups, v_field, T_field, labels, colors,
                     var_names=(r'$U$', r'$\hat{v}_1$', r'$\hat{T}_1$'),
                     title=None, figsize=(10, 6)):
    """Time series + v(x,t) and T(x,t) spatiotemporal fields."""
    xaxis = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    fig = plt.figure(figsize=figsize)
    spec = fig.add_gridspec(nrows=5, ncols=1)
    ax0 = None  # reference axis for sharing x
    panel_axes = []
    for row, (var_list, var_label) in enumerate(zip(data_groups, var_names)):
        ax = fig.add_subplot(spec[row, 0], sharex=ax0)
        if ax0 is None:
            ax0 = ax
        for i, data in enumerate(var_list):
            ax.plot(xaxis, data[sel0:sel1:interv], color=colors[i],
                    label=labels[i], linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylabel(var_label)
        plt.setp(ax.get_xticklabels(), visible=False)  # hide x labels except last
        if title is not None and row == 0:
            ax.set_title(title)
        panel_axes.append(ax)
    field_kw = dict(origin='lower', aspect='auto', vmin=-6, vmax=6, cmap='RdBu_r',
                    extent=[sel0 * dt, sel1 * dt, 0, 2 * np.pi])
    ax_v = fig.add_subplot(spec[3, 0], sharex=ax0)
    ax_v.imshow(v_field[sel0:sel1, :].T, **field_kw)
    ax_v.set_ylabel('$x$')
    plt.setp(ax_v.get_xticklabels(), visible=False)
    ax_T = fig.add_subplot(spec[4, 0], sharex=ax0)
    im = ax_T.imshow(T_field[sel0:sel1, :].T, **field_kw)
    ax_T.set_ylabel('$x$')
    ax_T.set_xlabel('$t$')
    fig.text(0.5, 1 - 2.26 / 5, r'$\mathbf{v(x,t)}$',
             fontweight='bold', va='center', ha='left')
    fig.text(0.5, 1 - 3. / 5, r'$\mathbf{T(x,t)}$',
             fontweight='bold', va='center', ha='left')
    panel_axes += [ax_v, ax_T]
    plt.tight_layout(rect=[0, 0.01, 1, 1], h_pad=0.42)
    fig.colorbar(im, ax=panel_axes, orientation='horizontal',
                 location='bottom', pad=0.14, fraction=0.01, aspect=40)


def plot_topobaro_forecast_comparison(dt, sel0, sel1, interv, xlim,
                                data_groups, v_fields, T_fields, labels, colors,
                                var_names=(r'$U$', r'$\hat{v}_1$', r'$\hat{T}_1$'),
                                line_width=1, title=None):
    """Time series + v(x,t) / T(x,t) fields for multiple forecast methods."""
    n_methods = len(v_fields)
    total_rows = 3 + 2 * n_methods
    time_axis = np.arange(sel0 * dt, sel1 * dt, interv * dt)

    fig, axes = plt.subplots(total_rows, 1,
                             figsize=(10, 1.1 * total_rows), sharex=True)
    # plt.subplots_adjust(hspace=0.01)

    legend_lines, legend_labels = [], []

    # Time series rows
    for i in range(3):
        ax = axes[i]
        for j, data in enumerate(data_groups[i]):
            line, = ax.plot(time_axis, data[sel0:sel1:interv],
                            color=colors[j], linewidth=line_width)
            if i == 0:
                legend_lines.append(line)
                legend_labels.append(labels[j])
        ax.set_ylabel(var_names[i])
        ax.set_xlim(xlim)

    # Field kwargs
    field_kw = dict(origin='lower', aspect='auto', vmin=-10, vmax=10, cmap='RdBu_r',
                    extent=[sel0 * dt, sel1 * dt, 0, 2 * np.pi])
    label_kw = dict(fontweight='bold', va='top', ha='right',
                    bbox=dict(facecolor='white', alpha=0.2, edgecolor='none'))

    for i in range(n_methods):
        ax = axes[3 + i]
        ax.imshow(v_fields[i][sel0:sel1].T, **field_kw)
        ax.set_ylabel(r'$x$')
        ax.text(0.995, 0.95, labels[i], transform=ax.transAxes, **label_kw)

    for i in range(n_methods):
        ax = axes[3 + n_methods + i]
        im_T = ax.imshow(T_fields[i][sel0:sel1].T, **field_kw)
        ax.set_ylabel(r'$x$')
        ax.text(0.995, 0.95, labels[i], transform=ax.transAxes, **label_kw)
        if i == n_methods - 1:
            ax.set_xlabel(r'$t$')

    fig.text(0.5, 1 - 3. / total_rows, r'$\mathbf{v(x,t)}$',
             fontweight='bold', va='center', ha='left')
    fig.text(0.5, 1 - 5.55 / total_rows, r'$\mathbf{T(x,t)}$',
             fontweight='bold', va='center', ha='left')
    if title is not None:
        fig.suptitle(title, y=0.96)
    fig.legend(legend_lines, legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels))
    plt.tight_layout(rect=[0, 0.01, 1, 0.96], pad=1.3, h_pad=0.7)
    fig.colorbar(im_T, ax=axes, location='bottom', pad=0.045, fraction=0.008, aspect=30)


def plot_pdf_and_joint(var1, var2, var_names, ylims, log=True, figsize=(9, 3)):
    """Marginal PDFs (with Gaussian fit) and joint PDF for two variables."""
    from scipy.stats import gaussian_kde, norm

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, (v, name) in enumerate(zip([var1, var2], var_names)):
        kde = gaussian_kde(v)
        x = np.linspace(v.min(), v.max(), 300)
        axes[i].plot(x, kde.evaluate(x), label='Truth', color='k')
        axes[i].plot(x, norm.pdf(x, v.mean(), v.std()), 'k--', label='Gaussian fit')
        # axes[i].set_title(f'PDF: {name}')
        axes[i].set_xlabel(f'{name}')
        axes[i].set_ylabel('density')
        axes[i].legend()
        axes[i].set_ylim(ylims[i])
        if log:
            axes[i].set_yscale('log')

    # Joint PDF
    xy = np.vstack([var1, var2])
    kde_joint = gaussian_kde(xy)
    X, Y = np.meshgrid(np.linspace(var1.min(), var1.max(), 100),
                        np.linspace(var2.min(), var2.max(), 100))
    Z = kde_joint(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    cf = axes[2].contourf(X, Y, Z, levels=50, cmap='binary')
    # axes[2].set_title(f'Joint PDF: ({var_names[0]}, {var_names[1]})')
    axes[2].set_xlabel(var_names[0])
    axes[2].set_ylabel(var_names[1])
    cbar = fig.colorbar(cf, ax=axes[2], fraction=0.03, pad=0.02, label='density')
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────

def plot_L63_clustering(
    Ls, dt_obs, lag, orig, lag_W,
    t_b, S_b, mem_nosel_b, mem_sel_b,
    xlim_b=(0, 200),
    figsize=(10.0, 3.8),
):
    """
    Combined L63 clustering figure.

    Panel (a): accuracy vs. temporal embedding length in physical time L * dt_obs.

    Panel (b): time-series comparison of FCM clustering with and without
    entropy-regularized feature selection against the true regime.
    """

    c_lag = 'dodgerblue'
    c_phi = 'r'
    c_true = 'k'

    acc_lag = np.nanmean(lag[:, :, 2], axis=1)
    acc_phi = np.nanmean(orig[:, :, 2], axis=1)

    L_best_lag = int(Ls[int(np.argmax(acc_lag))])
    L_best_phi = int(Ls[int(np.argmax(acc_phi))])

    tL = Ls * dt_obs
    tL_best_lag = L_best_lag * dt_obs
    tL_best_phi = L_best_phi * dt_obs

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2, 1,
        height_ratios=[1.0, 1.0],
        hspace=0.55,
        left=0.07, right=0.985, bottom=0.16, top=0.86
    )

    # ---- panel (a): window length sweep -----------------------------------
    ax_a = fig.add_subplot(gs[0, 0])

    ax_a.plot(
        tL, acc_lag, '-o',
        color=c_lag, lw=1.6, ms=4.5,
        label=r'Lag stacking'
    )
    ax_a.plot(
        tL, acc_phi, '-s',
        color=c_phi, lw=1.6, ms=4.5,
        label=r'Lag projection'
    )

    ax_a.axvline(tL_best_lag, color=c_lag, ls=':', lw=1.0, alpha=0.8)
    ax_a.axvline(tL_best_phi, color=c_phi, ls=':', lw=1.0, alpha=0.8)

    # ax_a.set_xlabel(r'Physical time $L\,\Delta t_{\mathrm{obs}}$')
    ax_a.set_ylabel('accuracy')
    ax_a.set_xticks(tL)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_title('(a)', loc='left', fontsize=11)

    ax_a.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        borderaxespad=0.2
    )

    # ---- panel (b): feature selection time series -------------------------
    ax_b = fig.add_subplot(gs[1, 0])

    m = (t_b >= xlim_b[0]) & (t_b <= xlim_b[1])
    ax_b.plot(
        t_b[m], S_b[m],
        color=c_true, lw=1.4,
        label='True regime'
    )
    ax_b.plot(
        t_b[m], mem_nosel_b[m], '--',
        color=c_lag, lw=1.2,
        label='Membership w/o feature selection'
    )
    ax_b.plot(
        t_b[m], mem_sel_b[m], '--',
        color=c_phi, lw=1.2,
        label='Membership w/ feature selection'
    )

    ax_b.set_ylim(-0.1, 1.1)
    ax_b.set_xlim(xlim_b)
    ax_b.set_xlabel(r'$t$')
    ax_b.set_ylabel('regime')
    ax_b.grid(True, alpha=0.3)
    ax_b.set_title('(b)', loc='left', fontsize=11)

    ax_b.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        borderaxespad=0.2
    )

    return fig


def plot_scatter_weights(score, trueLabels,
                         accuracy_entropy, accuracy_baseline,
                         W_entropy, W_baseline,
                         correct_entropy, correct_baseline, signalDim, figsize=(12, 5)):
    """PCA scatter plots and feature-weight bar charts for FCM clustering."""
    # Scatter
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    for ax, correct, acc, label in [
        (ax1, correct_entropy, accuracy_entropy, 'Entropy Reg.'),
        (ax2, correct_baseline, accuracy_baseline, 'No Entropy'),
    ]:
        ax.set_title(f'{label} (Acc: {acc * 100:.1f}%)')
        ax.grid(True)
        for cls, c in [(1, 'b'), (2, 'g')]:
            sel = correct & (trueLabels == cls)
            ax.scatter(score[sel, 0], score[sel, 1], c=c, label=f'Class {cls}', s=50)
        ax.scatter(score[~correct, 0], score[~correct, 1],
                   c='r', marker='x', label='Errors', s=50)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend()
    fig1.tight_layout()

    # Feature weights
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=figsize)
    for ax, W, label in [
        (ax3, W_entropy, 'Feature Weights with Entropy Regularization'),
        (ax4, W_baseline, 'Feature Weights without Entropy'),
    ]:
        ax.bar(np.arange(len(W)), W)
        ax.axvline(x=signalDim - 0.5, color='r', linestyle='--', linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Weight')
        ax.grid(True)
    fig2.tight_layout()


def plot_loss(loss_list):
    """Training loss curve."""
    plt.figure(figsize=(4, 3))
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()


def plot_grad_norms(grad_norm_log, figsize=(6, 4)):
    """Per-parameter gradient norms during optimisation."""
    fuzzy = np.array([float(g[0]) for g in grad_norm_log])
    center = np.array([float(g[1]) for g in grad_norm_log])
    W = np.array([float(g[2]) for g in grad_norm_log])
    steps = np.arange(len(grad_norm_log))

    plt.figure(figsize=figsize)
    plt.plot(steps, fuzzy, label='fuzzypartmat logits', linewidth=2)
    plt.plot(steps, center, label='centers', linewidth=2)
    plt.plot(steps, W, label='W logits', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm (L2)')
    plt.title('Gradient Norms During Optimization')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()


def plot_l63_regimes(dt, sel0, sel1, interv, S,
                     prior_weights1, prior_weights2, xlim, warmup=20, figsize=(10, 2.3)):
    """FCM clustering membership comparison for Lorenz-63."""
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    lines, labels = [], []

    l1, = ax.plot(time, S[sel0:sel1:interv], 'k', linewidth=1.5)
    l2, = ax.plot(time, prior_weights1[sel0:sel1:interv], 'b--', linewidth=1.5)
    l3, = ax.plot(time, prior_weights2[sel0:sel1:interv], 'r--', linewidth=1.5)
    lines.extend([l1, l2, l3])
    labels.extend(['True Regime',
                   'Membership w/o feature selection',
                   'Membership w/ feature selection'])

    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel('$t$')
    ax.set_title('FCM clustering with / without entropy-regularized feature selection')
    ax.set_xlim(xlim)

    fig.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.51, 0.12), ncol=3)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])


# ─────────────────────────────────────────────────────────────────────────────
# Data assimilation
# ─────────────────────────────────────────────────────────────────────────────

def plot_l63_da_series(dt, sel0, sel1, interv,
                    x_truth, y_truth, z_truth, S,
                    mean, spread, prior_weights, posterior_weights,
                    xlim, warmup=20,
                    prior_mean=None, prior_spread=None, obs=None, figsize=(10, 8)):
    """Lorenz-63 DA time series with regime panel."""
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    var_names = ['X', 'Y', 'Z']
    truth_vars = [x_truth, y_truth, z_truth]
    std = np.sqrt(spread)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    lines, labels = [], []

    for i in range(3):
        ax = axes[i]
        l1, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=1.5)
        l2, = ax.plot(time, mean[sel0:sel1:interv, i], 'r', linewidth=1.5)
        l3 = ax.fill_between(
            time,
            mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
            mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
            color='r', alpha=0.2)
        if prior_mean is not None:
            p_std = np.sqrt(prior_spread)
            ax.plot(time, prior_mean[sel0:sel1:interv, i], 'b', linewidth=1.5)
            ax.fill_between(
                time,
                prior_mean[sel0:sel1:interv, i] - p_std[sel0:sel1:interv, i],
                prior_mean[sel0:sel1:interv, i] + p_std[sel0:sel1:interv, i],
                color='b', alpha=0.2)
        if obs is not None:
            ax.plot(time, obs[sel0:sel1:interv, i], 'g', linewidth=1.5)

        ax.set_title(var_names[i])
        ax.set_xlim(xlim)
        if i == 0:
            lines.extend([l1, l2, l3])
            labels.extend(['Truth', 'Posterior Mean', 'Posterior Spread'])

        _add_corr_rmse_box(ax, truth_vars[i], mean[:, i], warmup)

    # Regime panel
    ax = axes[3]
    l4, = ax.plot(time, S[sel0:sel1:interv], 'k', linewidth=1.5)
    l5, = ax.plot(time, prior_weights[sel0:sel1:interv], 'b--', linewidth=1.5)
    l6, = ax.plot(time, posterior_weights[sel0:sel1:interv], 'r--', linewidth=1.5)
    lines.extend([l4, l5, l6])
    labels.extend(['True Regime', 'Prior Weight', 'Posterior Weight'])
    ax.set_ylim([-0.1, 1.1])
    ax.set_title('Regime')
    ax.set_xlim(xlim)

    fig.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.51, 0.04), ncol=6)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_L63_da_series_comparison(dt, sel0, sel1, interv, 
                                  series_list, series_labels,
                                  spread_list, S=None,
                                  prior_weights=None, posterior_weights=None,
                                  xlim=None, warmup=40, var_names=None,
                                  colors=('k', 'g', 'b', 'r'),
                                    line_width=1.5, title=None, figsize=(10, 6)):
    """Multi-method DA comparison for L63."""
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    t_idx = slice(sel0, sel1, interv)
    _, n_vars = series_list[0].shape
    n_rows = n_vars + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    lines, labels = [], []
    if title is not None:
        axes[0].set_title(title)

    # Signal rows
    for v_idx in range(n_vars):
        ax = axes[v_idx]        
        ax.set_ylabel(var_names[v_idx])
        for ms, ss, lbl, c in zip(series_list, spread_list, series_labels, colors):
            m = ms[t_idx, v_idx]
            l, = ax.plot(time, m, color=c, linewidth=line_width)
            if v_idx == 0:
                lines.append(l)
                labels.append(lbl)
            if ss is not None:
                s = ss[t_idx, v_idx]
                ax.fill_between(time, m - 2 * s, m + 2 * s, color=c, alpha=0.2)
        ax.set_xlim(xlim)
                
        _add_corr_rmse_box_multi(
            ax, series_list[0][:, v_idx],
            [m[:, v_idx] for m in series_list[1:]],
            colors[1:], warmup)

    # Regime panel
    ax = axes[-1]
    if S is not None:
        l, = ax.plot(time, S[t_idx], 'k--', linewidth=line_width)
        lines.append(l); labels.append('True Regime')
    if prior_weights is not None:
        l, = ax.plot(time, prior_weights[t_idx], '--', color='lightskyblue', linewidth=line_width)
        lines.append(l); labels.append('Prior Weight')
    if posterior_weights is not None:
        l, = ax.plot(time, posterior_weights[t_idx], 'r--', linewidth=line_width)
        lines.append(l); labels.append('Posterior Weight')
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel('regime')
    ax.set_xlabel('$t$')
    ax.set_xlim(xlim)

    fig.legend(lines, labels, loc='upper center', columnspacing=1,
               bbox_to_anchor=(0.52, 0.99), ncol=7)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_ou_da_series(dt, sel0, sel1, interv,
                   ur_truth, ui_truth, v_truth,
                   mean=None, spread=None,
                   prior_weights=None, posterior_weights=None,
                   xlim=None, warmup=20,
                   prior_mean=None, prior_spread=None,
                   obs=None, S=None,
                   var_names=('uR', 'uI', 'v'), figsize=(10, 8)):
    """Complex OU DA time series with regime panel."""
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    truth_vars = [ur_truth, ui_truth, v_truth]
    std = np.sqrt(spread)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    lines, labels = [], []

    for i in range(3):
        ax = axes[i]
        l1, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=1.5)
        l2, = ax.plot(time, mean[sel0:sel1:interv, i], 'r', linewidth=1.5)
        l3 = ax.fill_between(
            time,
            mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
            mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
            color='r', alpha=0.2)
        if prior_mean is not None:
            p_std = np.sqrt(prior_spread)
            ax.plot(time, prior_mean[sel0:sel1:interv, i], 'b', linewidth=1.5)
            ax.fill_between(
                time,
                prior_mean[sel0:sel1:interv, i] - p_std[sel0:sel1:interv, i],
                prior_mean[sel0:sel1:interv, i] + p_std[sel0:sel1:interv, i],
                color='b', alpha=0.2)
        if obs is not None:
            ax.plot(time, obs[sel0:sel1:interv, i], 'g', linewidth=1.5)

        ax.set_title(var_names[i])
        ax.set_xlim(xlim)
        if i == 0:
            lines.extend([l1, l2, l3])
            labels.extend(['Truth', 'Posterior Mean', 'Posterior Spread'])
        if i < 2:
            _add_corr_rmse_box(ax, truth_vars[i], mean[:, i], warmup)

    # Regime panel
    r_lines, r_labels = _regime_panel(
        axes[3], time[::1],
        S=S[sel0:sel1:interv] if S is not None else None,
        prior_weights=prior_weights[sel0:sel1:interv] if prior_weights is not None else None,
        posterior_weights=posterior_weights[sel0:sel1:interv] if posterior_weights is not None else None,
    )
    axes[3].set_xlim(xlim)
    lines.extend(r_lines)
    labels.extend(r_labels)

    fig.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.51, 0.04), ncol=6)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_ou_da_series_comparison(dt, sel0, sel1, interv, truth_vars,
                              mean=None, spread=None,
                              prior_weights=None, posterior_weights=None,
                              xlim=None, warmup=20,
                              prior_mean=None, prior_spread=None,
                              obs=None, S=None,
                              var_names=('Single-model EnKF',
                                         'Standard Multi-model EnKF',
                                         'Stochastic Parameterization EnKF',
                                         'Gaussian Mixture Multi-model EnKF',
                                         'Stochastic Parameterization EnKF: $v$',
                                         'Gaussian Mixture Multi-model EnKF: regime'),
                              y_labels=('Re[$u$]', 'Re[$u$]', 'Re[$u$]',
                                        'Re[$u$]', '$v$', 'weight'),
                              figsize=(10, 12)):
    """Multi-panel OU DA comparison with regime row."""
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    std = np.sqrt(spread)
    n_vars = len(truth_vars)

    fig, axes = plt.subplots(n_vars + 1, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [1] * (n_vars + 1)})
    lines, labels = [], []

    for i in range(n_vars):
        ax = axes[i]
        l1, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=1.5)
        l2, = ax.plot(time, mean[sel0:sel1:interv, i], 'r', linewidth=1.5)
        l3 = ax.fill_between(
            time,
            mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
            mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
            color='r', alpha=0.2)
        ax.set_ylabel(y_labels[i])
        if prior_mean is not None:
            p_std = np.sqrt(prior_spread)
            ax.plot(time, prior_mean[sel0:sel1:interv, i], 'b', linewidth=1.5)
            ax.fill_between(
                time,
                prior_mean[sel0:sel1:interv, i] - p_std[sel0:sel1:interv, i],
                prior_mean[sel0:sel1:interv, i] + p_std[sel0:sel1:interv, i],
                color='b', alpha=0.2)
        if obs is not None:
            ax.plot(time, obs[sel0:sel1:interv, i], 'g', linewidth=1.5)

        ax.set_title(var_names[i])
        ax.set_xlim(xlim)
        if i == 0:
            lines.extend([l1, l2, l3])
            labels.extend(['Truth', 'Posterior Mean', 'Posterior Spread'])
        if i < n_vars - 1:
            _add_corr_rmse_box(ax, truth_vars[i], mean[:, i], warmup)

    # Regime row
    ax = axes[-1]
    r_lines, r_labels = _regime_panel(
        ax, time,
        S=S[sel0:sel1:interv] if S is not None else None,
        prior_weights=prior_weights[sel0:sel1:interv] if prior_weights is not None else None,
        posterior_weights=posterior_weights[sel0:sel1:interv] if posterior_weights is not None else None,
    )
    ax.set_title(var_names[-1])
    ax.set_xlim(xlim)
    lines.extend(r_lines)
    labels.extend(r_labels)

    fig.legend(lines, labels, loc='upper center', columnspacing=1,
               bbox_to_anchor=(0.51, 0.04), ncol=6)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_topobaro_series_da(dt, sel0, sel1, interv, truth_vars,
                         mean=None, spread=None,
                         prior_weights=None, posterior_weights=None,
                         xlim=None, warmup=20,
                         prior_mean=None, prior_spread=None,
                         obs=None, S=None,
                         var_names=('$U$', '$v_1$', '$T_1$'), figsize=(10, 8)):
    """Topographic barotropic DA series with regime panel."""
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    std = np.sqrt(spread)

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    lines, labels = [], []

    for i in range(3):
        ax = axes[i]
        l1, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=1.5)
        l2, = ax.plot(time, mean[sel0:sel1:interv, i], 'r', linewidth=1.5)
        l3 = ax.fill_between(
            time,
            mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
            mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
            color='r', alpha=0.2)
        if prior_mean is not None:
            p_std = np.sqrt(prior_spread)
            ax.plot(time, prior_mean[sel0:sel1:interv, i], 'b', linewidth=1.5)
            ax.fill_between(
                time,
                prior_mean[sel0:sel1:interv, i] - p_std[sel0:sel1:interv, i],
                prior_mean[sel0:sel1:interv, i] + p_std[sel0:sel1:interv, i],
                color='b', alpha=0.2)
        if obs is not None:
            ax.plot(time, obs[sel0:sel1:interv, i], 'g', linewidth=1.5)

        ax.set_title(var_names[i])
        ax.set_xlim(xlim)
        if i == 0:
            lines.extend([l1, l2, l3])
            labels.extend(['Truth', 'Posterior Mean', 'Posterior Spread'])
        _add_corr_rmse_box(ax, truth_vars[i], mean[:, i], warmup)

    # Regime panel
    r_lines, r_labels = _regime_panel(
        axes[3], time,
        S=S[sel0:sel1:interv] if S is not None else None,
        prior_weights=prior_weights[sel0:sel1:interv] if prior_weights is not None else None,
        posterior_weights=posterior_weights[sel0:sel1:interv] if posterior_weights is not None else None,
    )
    axes[3].set_xlim(xlim)
    lines.extend(r_lines)
    labels.extend(r_labels)

    fig.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.51, 0.04), ncol=6)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_topobaro_series_comparison(dt, sel0, sel1, interv, truth_vars,
                                    means=None, spreads=None,
                                    prior_weights=None, posterior_weights=None,
                                    xlim=None, warmup=40, S=None,
                                    var_names=('$U$', '$v_1$', '$T_1$'),
                                    mean_labels=None,
                                    colors=('r', 'b', 'g', 'orange', 'purple', 'brown'),
                                    line_width=1.5, title=None, figsize=(10, 8)):
    """Multi-method topographic barotropic DA comparison."""
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    lines, labels = [], []
    if title is not None:
        axes[0].set_title(title)

    for i in range(3):
        ax = axes[i]
        l_truth, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=line_width)
        ax.set_ylabel(var_names[i])
        ax.set_xlim(xlim)
        if i == 0:
            lines.append(l_truth)
            labels.append('Truth')

        for m, lbl, c in zip(means, mean_labels, colors):
            l, = ax.plot(time, m[sel0:sel1:interv, i], color=c, linewidth=line_width)
            if i == 0:
                lines.append(l)
                labels.append(lbl)
        if spreads is not None:
            for std, m, c in zip(spreads, means, colors):
                ax.fill_between(
                    time,
                    m[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
                    m[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
                    color=c, alpha=0.2)

        _add_corr_rmse_box_multi(
            ax, truth_vars[i], [m[:, i] for m in means], list(colors), warmup,
            spacing_x=0.058, spacing_y=0.1)

    # Regime panel
    ax = axes[3]
    if S is not None:
        l, = ax.plot(time, S[sel0:sel1:interv], 'k', linewidth=line_width)
        lines.append(l); labels.append('True Regime')
    if prior_weights is not None:
        l, = ax.plot(time, prior_weights[sel0:sel1:interv], 'b--', linewidth=line_width)
        lines.append(l); labels.append('Prior Weight')
    if posterior_weights is not None:
        l, = ax.plot(time, posterior_weights[sel0:sel1:interv], 'r--', linewidth=line_width)
        lines.append(l); labels.append('Posterior Weight')
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel('weight')
    ax.set_xlabel('$t$')
    ax.set_xlim(xlim)

    fig.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.51, 0.04), ncol=6)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])


def plot_topobaro_da_fields_comparison(dt, sel0, sel1,
                                    v_fields, T_fields, method_labels, vlim=6, figsize=(5.6, 9)):
    """Side-by-side v(x,t) and T(x,t) fields for multiple methods (Hovmöller style)."""
    n_methods = len(method_labels)
    n_vars = 2
    extent = [0, 2 * np.pi, sel0 * dt, sel1 * dt]

    fig, axes = plt.subplots(
        nrows=n_vars, ncols=n_methods,
        figsize=figsize,
        sharex=True, sharey=True, constrained_layout=True)
    fig.get_layout_engine().set(wspace=0.15)

    if n_methods == 1:
        axes = np.expand_dims(axes, axis=1)

    field_kw = dict(origin='lower', aspect='auto', vmin=-vlim, vmax=vlim,
                    cmap='RdBu_r', extent=extent)
    var_names = [r'$v(x,t)$', r'$T(x,t)$']
    all_fields = [v_fields, T_fields]

    pcm_row = [None] * n_vars
    for v_idx in range(n_vars):
        for c_idx in range(n_methods):
            ax = axes[v_idx, c_idx]
            C = all_fields[v_idx][c_idx][sel0:sel1, :]
            pcm_row[v_idx] = ax.imshow(C, **field_kw)
            if v_idx == 0:
                ax.set_title(method_labels[c_idx], pad=4)
            ax.set_ylabel(r'$t$' if c_idx == 0 else '')
            if v_idx == n_vars - 1:
                ax.set_xlabel(r'$x$')

    for v_idx in range(n_vars):
        cb = fig.colorbar(pcm_row[v_idx], ax=axes[v_idx, -1],
                          fraction=0.08, pad=0.15, aspect=25)
        cb.set_label(var_names[v_idx])


def plot_gmm_pdf_3vars(means_list, stds_list, weights_list,
                       x_ranges=None, num_points=1000,
                       var_names=('X', 'Y', 'Z'), figsize=(10, 3)):
    """1-D GMM PDFs for three variables."""
    from scipy.stats import norm

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for i in range(3):
        means = np.asarray(means_list[i])
        stds = np.asarray(stds_list[i])
        weights = np.asarray(weights_list)

        if x_ranges is None or x_ranges[i] is None:
            xmin, xmax = np.min(means - 4 * stds), np.max(means + 4 * stds)
        else:
            xmin, xmax = x_ranges[i]

        x = np.linspace(xmin, xmax, num_points)
        pdf = sum(w * norm.pdf(x, loc=mu, scale=sig)
                  for mu, sig, w in zip(means, stds, weights))

        ax = axes[i]
        ax.plot(x, pdf, 'k')
        for j, mu in enumerate(means):
            ax.axvline(mu, color='k', alpha=0.5,
                       label=rf'$\mu_{j}={mu:.2f}$')
        ax.set_xlabel(var_names[i])
        ax.legend()
    axes[0].set_ylabel('Density')
    plt.tight_layout()


def plot_gmm_pdf_3vars_multi(means_array, stds_array, weights_array,
                             x_ranges=None, num_points=1000,
                             var_names=('X', 'Y', 'Z'),
                             labels=None, colors=None,
                             title=None, truth_array=None, figsize=(10, 3)):
    """Overlaid multi-GMM PDFs for three variables."""
    from scipy.stats import norm

    K = means_array.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    legend_lines = []
    if labels is None:
        labels = [f'GMM {i}' for i in range(K)]
    if colors is None:
        colors = ['k', 'r', 'b'][:K]

    for i in range(3):
        ax = axes[i]
        for k in range(K):
            m, s, w = means_array[k, i], stds_array[k, i], weights_array[k, i]
            mask = w > 0
            m, s, w = m[mask], s[mask], w[mask]

            if x_ranges is None or x_ranges[i] is None:
                xmin, xmax = np.min(m - 4 * s), np.max(m + 4 * s)
            else:
                xmin, xmax = x_ranges[i]

            x = np.linspace(xmin, xmax, num_points)
            pdf = sum(wj * norm.pdf(x, loc=mj, scale=sj)
                      for mj, sj, wj in zip(m, s, w))
            line, = ax.plot(x, pdf, color=colors[k], linewidth=2)
            for mu in m:
                ax.axvline(mu, ls='--', color=colors[k], alpha=0.3)
            if i == 0:
                legend_lines.append(line)

        line = ax.axvline(truth_array[i], color=colors[K], alpha=0.8)
        if i == 0:
            legend_lines.append(line)
        ax.set_xlabel(var_names[i])

    axes[0].set_ylabel('density')
    fig.suptitle(title)
    fig.legend(legend_lines, labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.12), ncol=K + 1)
    plt.tight_layout(rect=[0, 0.06, 1, 1])


def plot_series(dt, sel0, sel1, interv, truth_vars,
                mean=None, spread=None,
                prior_weights=None, posterior_weights=None,
                xlim=None, warmup=20,
                prior_mean=None, prior_spread=None,
                obs=None, S=None,
                var_names=('$U$', '$v_1$', '$T_1$')):
    """Generic DA time series (N variables) with regime panel."""
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    std = np.sqrt(spread)
    nvars = len(truth_vars)

    fig, axes = plt.subplots(nvars + 1, 1, figsize=(10, 2 * (nvars + 1)),
                             sharex=True,
                             gridspec_kw={'height_ratios': [1] * (nvars + 1)})
    lines, labels = [], []

    for i in range(nvars):
        ax = axes[i]
        l1, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=1.5)
        l2, = ax.plot(time, mean[sel0:sel1:interv, i], 'r', linewidth=1.5)
        l3 = ax.fill_between(
            time,
            mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
            mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
            color='r', alpha=0.2)
        if prior_mean is not None:
            p_std = np.sqrt(prior_spread)
            ax.plot(time, prior_mean[sel0:sel1:interv, i], 'b', linewidth=1.5)
            ax.fill_between(
                time,
                prior_mean[sel0:sel1:interv, i] - p_std[sel0:sel1:interv, i],
                prior_mean[sel0:sel1:interv, i] + p_std[sel0:sel1:interv, i],
                color='b', alpha=0.2)
        if obs is not None:
            ax.plot(time, obs[sel0:sel1:interv, i], 'g', linewidth=1.5)

        ax.set_title(var_names[i])
        ax.set_xlim(xlim)
        if i == 0:
            lines.extend([l1, l2, l3])
            labels.extend(['Truth', 'Posterior Mean', 'Posterior Spread'])
        _add_corr_rmse_box(ax, truth_vars[i], mean[:, i], warmup)

    # Regime panel
    r_lines, r_labels = _regime_panel(
        axes[nvars], time,
        S=S[sel0:sel1:interv] if S is not None else None,
        prior_weights=prior_weights[sel0:sel1:interv] if prior_weights is not None else None,
        posterior_weights=posterior_weights[sel0:sel1:interv] if posterior_weights is not None else None,
    )
    axes[nvars].set_xlim(xlim)
    lines.extend(r_lines)
    labels.extend(r_labels)

    fig.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.51, 0.04), ncol=6)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# ─────────────────────────────────────────────────────────────────────────────
# Model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_histograms_univar(hist_data, variables=('x', 'y', 'z'), figsize=(12, 4)):
    """Per-variable, per-regime truth vs model histogram comparison."""
    n_regimes = len(hist_data[variables[0]])
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, n_regimes, figsize=figsize, squeeze=False)

    for i in range(n_regimes):
        for j, var in enumerate(variables):
            p_hat, q_hat, bin_edges, regime_id, model_id = hist_data[var][i]
            ax = axes[j][i]
            edges = bin_edges[0]
            centres = 0.5 * (edges[:-1] + edges[1:])
            w = (edges[1] - edges[0]) * 0.4
            ax.bar(centres - w / 2, p_hat, width=w, color='k', label='Truth', alpha=0.7)
            ax.bar(centres + w / 2, q_hat, width=w, color='r', label='Model', alpha=0.7)
            ax.set_title(f'Regime {regime_id}, Model {model_id}')
            if i == 0:
                ax.set_ylabel(var)
            if i == 0 and j == 0:
                ax.legend()
    fig.tight_layout()
    return fig


def plot_histogram_comparison(p_hat, q_hat, bin_edges, title='', var_name='x', figsize=(6, 4)):
    """Single-variable truth vs model histogram."""
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w = (bin_edges[1] - bin_edges[0]) * 0.4

    plt.figure(figsize=figsize)
    plt.bar(centres - w / 2, p_hat, width=w, color='k', label='Truth', alpha=0.7)
    plt.bar(centres + w / 2, q_hat, width=w, color='r', label='Model', alpha=0.7)
    plt.xlabel(var_name)
    plt.ylabel('Probability')
    plt.title(title or f'Histogram comparison for {var_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()


def plot_3d_histogram(p, edges, title='3D Histogram', threshold=0.001, figsize=(5, 5)):
    """3-D voxel representation of a joint histogram."""
    filled = p > threshold
    values = p[filled] / p.max()

    centres = [0.5 * (e[:-1] + e[1:]) for e in edges]
    X, Y, Z = np.meshgrid(*centres, indexing='ij')
    size = edges[0][1] - edges[0][0]
    fc = plt.cm.gist_gray(values)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    for x, y, z, c in zip(X[filled], Y[filled], Z[filled], fc):
        ax.bar3d(x, y, z, size, size, size, color=c, edgecolor='k', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()


def plot_all_histograms_univar_from_hist_pervar(
        hist_pervar, var_indices, model_ids=None, var_labels=None,
        figsize=(12, 4), model_names=None, model_colors=None):
    """Multi-model per-variable histograms from ``hist_pervar`` structure.
    """
    n_vars_total = len(hist_pervar)
    if var_labels is None:
        var_labels = [f'var {v}' for v in range(n_vars_total)]
    regime_ids_set, model_ids_set = set(), set()
    for v in range(n_vars_total):
        for (p_v, q_v, edges_v, rid, mid, _) in hist_pervar[v]:
            if p_v is not None and q_v is not None and edges_v is not None:
                regime_ids_set.add(rid)
                model_ids_set.add(mid)
    if model_ids is None:
        model_ids = sorted(model_ids_set)
    else:
        model_ids = list(model_ids)
    regime_ids = sorted(regime_ids_set)
    n_regimes = len(regime_ids)
    n_models_plot = len(model_ids)
    n_vars_plot = len(var_indices)
    
    fig, axes = plt.subplots(n_vars_plot, n_regimes, figsize=figsize, squeeze=False)

    # Build lookup
    lookup = {}
    for v in range(n_vars_total):
        for (p_v, q_v, edges_v, rid, mid, _) in hist_pervar[v]:
            lookup[(v, rid, mid)] = (p_v, q_v, edges_v)
    var_minmax = {}
    var_span = {}
    var_center = {}
    for v in var_indices:
        mins, maxs = [], []
        for rid in regime_ids:
            for mid in model_ids:
                if (v, rid, mid) not in lookup:
                    continue
                p_v, q_v, edges_v = lookup[(v, rid, mid)]
                if edges_v is None:
                    continue
                e = np.asarray(edges_v)
                if e.ndim != 1 or e.size < 2:
                    continue
                mins.append(float(np.min(e)))
                maxs.append(float(np.max(e)))
        if mins and maxs:
            mn, mx = min(mins), max(maxs)
            var_minmax[v] = (mn, mx)
            var_span[v] = mx - mn
            var_center[v] = 0.5 * (mn + mx)
    # Choose a common span (median across plotted variables that have data)
    spans = [var_span[v] for v in var_indices if v in var_span and var_span[v] > 0]
    common_span = np.median(spans) if spans else None
    # Build per-variable xlim: same for all regimes in that row.
    # Centered at each variable's own center, span forced to common_span.
    xlims_by_var = {}
    if common_span is not None and common_span > 0:
        for v in var_indices:
            if v not in var_center:
                continue
            c = var_center[v]
            half = 0.5 * common_span
            xlims_by_var[v] = (c - half, c + half)
    else:
        # Fallback: just use each variable's min/max
        for v in var_indices:
            if v in var_minmax:
                xlims_by_var[v] = var_minmax[v]

    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    legend_ax = None

    for row, v in enumerate(var_indices):
        for col, rid in enumerate(regime_ids):
            ax = axes[row, col]

            entries = [(mid, *lookup[(v, rid, mid)])
                       for mid in model_ids if (v, rid, mid) in lookup]
            if not entries:
                ax.set_visible(False)
                continue

            mid0, p0, q0, e0 = entries[0]
            edges = np.asarray(e0)
            if edges.ndim != 1 or edges.size < 2:
                ax.set_visible(False)
                continue

            centres = 0.5 * (edges[:-1] + edges[1:])
            bw = edges[1] - edges[0]

            truth_label = 'Truth' if legend_ax is None else None
            ax.bar(centres, p0, width=0.4 * bw, color='k', alpha=0.7, label=truth_label)

            mbw = 0.6 * bw / max(n_models_plot, 1)
            offsets = np.linspace(-0.225 * bw, 0.225 * bw, n_models_plot)
            for idx_m, (mid, _, q_m, _) in enumerate(entries):
                if q_m is None:
                    continue
                if model_colors is not None:
                    c = model_colors[idx_m]
                else:
                    c = cycle_colors[idx_m % len(cycle_colors)]
                lbl = (model_names[idx_m] if model_names else f'Model {mid}') if legend_ax is None else None
                ax.bar(centres + offsets[idx_m], q_m, width=mbw,
                       color=c, alpha=0.7, label=lbl)

            # Apply unified x-axis per variable (row)
            if v in xlims_by_var:
                ax.set_xlim(*xlims_by_var[v])
            if row == 0:
                ax.set_title(f'Regime {rid}')
            if col == 0:
                # Use the correct variable label (v), not row index
                ax.set_ylabel(var_labels[v] if v < len(var_labels) else f'var {v}')
            if legend_ax is None:
                legend_ax = ax

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    if legend_ax is not None:
        h, l = legend_ax.get_legend_handles_labels()
        if h:
            fig.legend(h, l, loc='lower center',
                       bbox_to_anchor=(0.5, 0.04), ncol=len(l))
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Schematic
# ─────────────────────────────────────────────────────────────────────────────

def plot_vertical_gaussian_mixture(
        components=((0.1, -0.6, 0.4), (0.3, 1.2, 1.2), (0.6, 1.6, 0.5)),
        x_range=(-5, 5), resolution=1000,
        highlight_idx=None, figsize=(2, 2), color='r'):
    """Vertical Gaussian mixture schematic (PDF on x-axis, value on y-axis)."""
    from scipy.stats import norm

    x = np.linspace(*x_range, resolution)
    y_comp = [w * norm.pdf(x, mu, sig) for w, mu, sig in components]
    y_mix = sum(y_comp)

    plt.figure(figsize=figsize)
    curve = y_comp[highlight_idx] if highlight_idx is not None else y_mix
    plt.plot(curve, x, color, linewidth=4.5)
    plt.ylim(*x_range)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


# ─────────────────────────────────────────────────────────────────────────────
# ENSO
# ─────────────────────────────────────────────────────────────────────────────

def plot_eofs(eofs, evr, modes=(1, 2, 3), vmin=None, vmax=None):
    """EOF patterns with shared colorbar."""
    modes = list(modes)
    fig, axs = plt.subplots(len(modes), 1,
                            figsize=(6, 2 * len(modes)), constrained_layout=True)
    if len(modes) == 1:
        axs = [axs]

    for ax, m in zip(axs, modes):
        patt = eofs.sel(mode=m).sortby('lat')
        im = patt.plot.imshow(ax=ax, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                              add_colorbar=False)
        ax.set_title('')
        ax.text(0.02, 0.02, f'EOF {m} ({evr.sel(mode=m).item() * 100:.1f}% var)',
                transform=ax.transAxes, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))
        ax.set_xlabel('Longitude' if ax is axs[-1] else '')
        ax.set_ylabel('Latitude')

    fig.colorbar(im, ax=axs, orientation='vertical',
                 aspect=50, shrink=0.8, fraction=0.1, pad=0.02)
    fig.suptitle('EOFs')


def plot_pcs(pcs, modes=(1, 2, 3)):
    """Principal component time series."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    for m in modes:
        ax.plot(pcs['time'].values, pcs.sel(mode=m), label=f'PC{m}')
    ax.legend()
    ax.grid(True, alpha=0.35)
    ax.set_xlabel('Time')
    ax.set_ylabel('PC (arb. units)')
    plt.tight_layout()


def plot_nino34_with_regimes(nino34, labels, time=None,
                             regime_names=None, colors=None, rolling=None,
                             title='Niño 3.4 with clustered regimes'):
    """Niño-3.4 index coloured by cluster regime."""
    from matplotlib.lines import Line2D

    nino34 = np.asarray(nino34)
    labels = np.asarray(labels)
    assert nino34.shape == labels.shape
    Nt = nino34.size

    x = np.arange(Nt) if time is None else np.asarray(time)
    xlabel = 'Time index' if time is None else 'Time'

    good = np.isfinite(nino34) & np.isfinite(labels)
    x_plot, y_plot = x[good], nino34[good]
    lab_plot = labels[good].astype(int)
    uniq = np.unique(lab_plot)

    assert len(colors) >= len(uniq)
    palette = colors[:len(uniq)]
    color_map = {lab: palette[i] for i, lab in enumerate(uniq)}
    name_map = {lab: (regime_names[lab] if lab >= 0 else 'Noise') for lab in uniq}

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x, nino34, 'k', lw=0.9, alpha=0.6)
    for lab in uniq:
        sel = lab_plot == lab
        ax.scatter(x_plot[sel], y_plot[sel], s=14,
                   color=color_map[lab], label=name_map[lab], zorder=3)

    ax.set_ylabel('Niño index')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Deduplicate legend
    handles, labs_ = ax.get_legend_handles_labels()
    seen, new_h, new_l = {}, [], []
    for h, l in zip(handles, labs_):
        if l not in seen:
            seen[l] = True
            new_h.append(h)
            new_l.append(l)
    ax.legend(new_h, new_l, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=len(new_l))

    plt.tight_layout()
    return fig, ax


def plot_regime_means(mean_maps, freq, regimes=None, cmap='RdBu_r'):
    """Regime-mean SSTA maps with shared symmetric colour scale."""
    if regimes is None:
        regimes = mean_maps.regime.values
    sel = mean_maps.sel(regime=regimes)
    vmax = float(np.nanmax(np.abs(sel.values)))

    fig, axs = plt.subplots(len(regimes), 1,
                            figsize=(6, 1.5 * len(regimes)),
                            sharex=True, constrained_layout=True)
    if len(regimes) == 1:
        axs = [axs]

    frq_map = {int(r): float(freq.sel(regime=r).item()) for r in freq.regime.values}

    for ax, reg in zip(axs, regimes):
        im = sel.sel(regime=reg).plot.imshow(
            ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax, add_colorbar=False)
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude' if ax is axs[-1] else '')
        ax.set_title('')
        frq = frq_map.get(int(reg), 0.0) * 100
        ax.text(0.02, 0.02, f'Regime {reg} ({frq:.1f}%)',
                transform=ax.transAxes, ha='left', va='bottom',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))

    fig.colorbar(im, ax=axs, orientation='vertical',
                 aspect=50, shrink=0.8, fraction=0.1, pad=0.02)
    fig.suptitle('Mean SSTA by regime')


def plot_hovmoller_with_regime(sst_eq_1d, time_vals, cluster_2d, cluster_labels,
                               cmap_regimes, title='SSTA', vmin=-1, vmax=1, figsize=(2.0, 6.5)):
    """Equatorial Hovmöller diagram with adjacent regime strip."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.dates as mdates

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.15)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)

    # Hovmöller
    sst = sst_eq_1d.transpose('time', 'lon')
    pcm = ax1.pcolormesh(sst['lon'].values, time_vals, sst.values,
                         cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
    ax1.set_xlabel('Lon (°E)')
    ax1.set_ylabel('year')
    ax1.set_title(title)
    ax1.yaxis.set_major_locator(mdates.YearLocator(5))
    ax1.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.set_xticks([130, 180, 230, 280])
    ax1.set_xlim(130, 280)

    cax = inset_axes(ax1, width='85%', height='1.5%', loc='lower center',
                     bbox_to_anchor=(0, -0.12, 1, 1),
                     bbox_transform=ax1.transAxes, borderpad=0)
    fig.colorbar(pcm, cax=cax, orientation='horizontal')

    # Regime strip
    ax2.imshow(cluster_2d, aspect='auto', cmap=cmap_regimes, origin='lower',
               extent=[0, 1, mdates.date2num(time_vals[0]),
                       mdates.date2num(time_vals[-1])])
    ax2.set_xticks([])
    ax2.set_title('regime')
    ax2.tick_params(labelleft=False)

    uniq = np.unique(cluster_labels)
    handles = [plt.Line2D([0], [0], marker='s', ls='',
                          markersize=6, markerfacecolor=cmap_regimes(i),
                          markeredgecolor='none')
               for i in uniq]
    ax2.legend(handles, [f'{i}' for i in uniq],
               loc='lower center', bbox_to_anchor=(0.6, -0.15), ncol=1,
               frameon=False, columnspacing=0.6, handletextpad=0.3,
               borderpad=0.2, labelspacing=0.2, handlelength=0.8)

    fig.subplots_adjust(bottom=0.14, top=0.96, left=0.23, right=0.9)
    return fig, (ax1, ax2)


def hovmoller_compare(data_by_col, time_by_col, lon, var_names,
                      vlims=None, yr_locator=1, figsize=(6, 9)):
    """Multi-column Hovmöller comparison with shared colorbars per variable."""
    import matplotlib.dates as mdates

    col_order = list(data_by_col.keys())
    n_cols = len(col_order)
    n_vars = len(var_names)

    def _hovmoller(ax, Xlon, Ytime, C, vmin, vmax, title=None):
        pcm = ax.pcolormesh(Xlon, Ytime, C,
                            cmap='RdBu_r', shading='auto', vmin=vmin, vmax=vmax)
        if title:
            ax.set_title(title, pad=4)
        ax.set_xlim(Xlon.min(), Xlon.max())
        ax.set_xticks([130, 180, 230, 280])
        ax.yaxis.set_major_locator(mdates.YearLocator(yr_locator))
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        return pcm

    fig, axes = plt.subplots(
        nrows=n_vars, ncols=n_cols,
        figsize=figsize,
        sharex=True, sharey=True, constrained_layout=True,
        )
    # fig.get_layout_engine().set(wspace=0.001, hspace=0.001)

    if n_vars == 1:
        axes = np.expand_dims(axes, axis=0)

    pcm_row = [None] * n_vars
    for v_idx, vname in enumerate(var_names):
        vmin, vmax = vlims[v_idx]
        for c_idx, col_name in enumerate(col_order):
            ax = axes[v_idx, c_idx]
            C = data_by_col[col_name][:, v_idx, :]
            pcm_row[v_idx] = _hovmoller(
                ax, lon, time_by_col[col_name], C, vmin, vmax,
                title=col_name if v_idx == 0 else None)
            ax.set_ylabel('year' if c_idx == 0 else '')
            if v_idx == n_vars - 1:
                ax.set_xlabel('Lon (°E)')

    for v_idx, vname in enumerate(var_names):
        cb = fig.colorbar(pcm_row[v_idx], ax=axes[v_idx, -1],
                          fraction=0.08, pad=0.02, aspect=25)
        cb.set_label(vname)
        
    return fig


def plot_enso_da_series_and_weights(
        time, series_list, spread_list=None, series_labels=None,
        weights_list=None, weights_ylabels=None, weights_rowlabels_list=None,
        sel0=0, sel1=None, interv=1, warmup=24,
        var_names=None,
        colors=('k', 'g', 'b', 'r', 'orange', 'purple', 'brown'),
        line_width=1.5, title=None, yr_locator=2, figsize=(10, 13.5)):
    """ENSO DA signal rows + optional weight-heatmap rows."""
    import matplotlib.dates as mdates

    time = np.asarray(time)
    T = time.shape[0]
    if sel1 is None:
        sel1 = T
    t_idx = slice(sel0, sel1, interv)
    time_sel = time[t_idx]

    n_series = len(series_list)
    series_list = [np.asarray(s) for s in series_list]
    if spread_list is None:
        spread_list = [None] * n_series
    _, n_vars = series_list[0].shape
    colors = list(colors)

    if weights_list is None:
        weights_list = []
    n_weight_rows = len(weights_list)
    if weights_rowlabels_list is None:
        weights_rowlabels_list = [None] * n_weight_rows

    n_rows = n_vars + n_weight_rows
    fig, axes = plt.subplots(
        n_rows, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [1.0] * n_rows})
    if n_rows == 1:
        axes = np.array([axes])
    lines, labels = [], []
    if title is not None:
        axes[0].set_title(title)

    # Signal rows
    for v_idx in range(n_vars):
        ax = axes[v_idx]
        ax.set_ylabel(var_names[v_idx])
        for ms, ss, lbl, c in zip(series_list, spread_list, series_labels, colors):
            m = ms[t_idx, v_idx]
            l, = ax.plot(time_sel, m, color=c, linewidth=line_width)
            if v_idx == 0:
                lines.append(l)
                labels.append(lbl)
            if ss is not None:
                s = ss[t_idx, v_idx]
                ax.fill_between(time_sel, m - 2 * s, m + 2 * s, color=c, alpha=0.2)

        _add_corr_rmse_box_multi(
            ax, series_list[0][:, v_idx],
            [m[:, v_idx] for m in series_list[1:]],
            colors[1:], warmup)

    # Weight heatmap rows
    im_for_cbar = None
    for w_idx, (W, ylabel, ticklabels) in enumerate(
            zip(weights_list, weights_ylabels, weights_rowlabels_list)):
        ax_w = axes[n_vars + w_idx]
        W = np.asarray(W)
        W_sel = W[t_idx]
        _, K = W_sel.shape
        if ticklabels is None:
            ticklabels = [f'{k}' for k in range(K)]

        t0 = mdates.date2num(time_sel[0])
        t1 = mdates.date2num(time_sel[-1])
        im = ax_w.imshow(
            W_sel.T, cmap='inferno', aspect='auto', origin='lower',
            interpolation='nearest', vmin=0.0, vmax=1.0,
            extent=[t0, t1, -0.5, K - 0.5])
        ax_w.set_yticks(np.arange(K))
        ax_w.set_yticklabels(ticklabels)
        if ylabel is not None:
            ax_w.set_ylabel(ylabel)
        ax_w.xaxis_date()
        ax_w.xaxis.set_major_locator(mdates.YearLocator(yr_locator))
        ax_w.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        im_for_cbar = im

    axes[-1].set_xlabel('year')
    fig.legend(lines, labels, loc='upper center',
               bbox_to_anchor=(0.52, 0.99), ncol=min(6, len(labels)))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.945, hspace=0.1)

    if im_for_cbar is not None:
        cax = fig.add_axes([0.37, 0.03, 0.3, 0.008])
        cbar = fig.colorbar(im_for_cbar, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)

    return fig
