import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

################################################### Model #####################################################    
def plot_trajectories(x_truth, y_truth, z_truth, dt, sel0=10000, sel1=20000, interv=10):
    xaxis = np.arange(sel0 * dt, sel1 * dt, interv * dt)

    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    gs0 = GridSpec(1, 2, figure=fig)

    # Left column: Individual trajectories
    gs00 = gs0[0].subgridspec(3, 6)
    ax1 = fig.add_subplot(gs00[0, :])
    ax2 = fig.add_subplot(gs00[1, :])
    ax3 = fig.add_subplot(gs00[2, :])

    ax1.plot(xaxis, x_truth[sel0:sel1:interv])
    ax1.set_xlim(xaxis[0], xaxis[-1])
    ax1.set_title('(a) Sample trajectory of x')

    ax2.plot(xaxis, y_truth[sel0:sel1:interv])
    ax2.set_xlim(xaxis[0], xaxis[-1])
    ax2.set_title('(b) Sample trajectory of y')

    ax3.plot(xaxis, z_truth[sel0:sel1:interv])
    ax3.set_xlim(xaxis[0], xaxis[-1])
    ax3.set_title('(c) Sample trajectory of z')
    ax3.set_xlabel('t')

    # Right column: 2D and 3D trajectories
    gs01 = gs0[1].subgridspec(2, 7)
    ax4 = fig.add_subplot(gs01[0, :3])
    ax5 = fig.add_subplot(gs01[0, 4:])
    ax6 = fig.add_subplot(gs01[1, :3])
    ax7 = fig.add_subplot(gs01[1, 4:], projection='3d')

    ax4.plot(x_truth, y_truth, lw=0.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('(d) 2D trajectory of x and y')

    ax5.plot(y_truth, z_truth, lw=0.5)
    ax5.set_xlabel('y')
    ax5.set_ylabel('z')
    ax5.set_title('(e) 2D trajectory of y and z')

    ax6.plot(z_truth, x_truth, lw=0.5)
    ax6.set_xlabel('z')
    ax6.set_ylabel('x')
    ax6.set_title('(f) 2D trajectory of z and x')

    ax7.plot(x_truth, y_truth, z_truth, lw=0.5)
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_zlabel('z')
    ax7.set_title('(g) 3D trajectory of x, y, and z')
    ax7.grid(False)
    plt.tight_layout()

def plot_ou_series_pdf_acf(dt, sel0, sel1, interv, xlim,
                           ur_list, ui_list, v_list, labels, colors, max_lag=4000):
    """
    Plot time series, PDFs (log-scale), and ACFs for ur, ui, and v.
    Layout: 3 rows (variables), 3 columns (time series | PDF | ACF)
    
    Parameters:
    - *_list: list of arrays of shape (T,)
    - sel0, sel1: time index range
    - interv: interval for time series downsampling
    - xlim: x-axis limits for time plots
    - ylim: x-axis limits for PDF plots (log x-scale)
    """
    from scipy.stats import gaussian_kde, norm
    from statsmodels.tsa.stattools import acf

    xaxis = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    lag_axis = dt * np.arange(max_lag + 1)

    fig = plt.figure(figsize=(10, 5))
    widths = [5, .8, .8]
    heights = [1, 1, 1]
    spec = fig.add_gridspec(nrows=3, ncols=3, width_ratios=widths, height_ratios=heights)
    plt.subplots_adjust(wspace=0.5, hspace=0.6)

    var_names = [r'$u_R$', r'$u_I$', r'$v$']
    data_groups = [ur_list, ui_list, v_list]
    legend_lines = []
    legend_labels = []

    for row, (var_list, var_label) in enumerate(zip(data_groups, var_names)):
        ax_ts  = fig.add_subplot(spec[row, 0])  # Time series
        ax_pdf = fig.add_subplot(spec[row, 1])  # PDF
        ax_acf = fig.add_subplot(spec[row, 2])  # ACF

        for i, data in enumerate(var_list):
            samples = data[sel0:sel1]
            series = data[sel0:sel1:interv]

            # --- Time Series ---
            line_model, = ax_ts.plot(xaxis, series, color=colors[i], label=labels[i])
            if row == 0 and i == 0:  # only track lines from the first row to avoid duplicates
                legend_lines.append(line_model)
                legend_labels.append(labels[i])
        
            # --- PDF ---
            kde = gaussian_kde(samples)
            x_pdf = np.linspace(samples.min(), samples.max(), 300)
            p_pdf = kde.evaluate(x_pdf)
            ax_pdf.plot(p_pdf, x_pdf, color=colors[i])
            if i == 0:
                mean, std = samples.mean(), samples.std()
                gauss = norm.pdf(x_pdf, mean, std)
                line_gauss, = ax_pdf.plot(gauss, x_pdf, 'k--', label='Gaussian fit')
                if row == 0:
                    legend_lines.append(line_gauss)
                    legend_labels.append('Gaussian fit')
                
            # --- ACF ---
            acf_vals = acf(samples, nlags=max_lag, fft=True)
            ax_acf.plot(lag_axis, acf_vals, color=colors[i])

        # Axis labels/titles
        ax_ts.set_xlim(xlim)
        ax_ts.set_ylabel(var_label)
        ax_ts.set_xlabel("t")
        if row == 0:
            ax_ts.set_title("Time Series")

        # ax_pdf.set_xscale("log", base=10)
        if row == 0:
            ax_pdf.set_title("PDF")

        ax_acf.set_xlim(0, lag_axis[-1])
        # ax_acf.set_xlabel("lag (t)")
        if row == 0:
            ax_acf.set_title("ACF")

    # Global legend
    fig.legend(legend_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=len(legend_labels), fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

def plot_baro_series_pdf_acf(dt, sel0, sel1, interv, xlim,
                           data_groups, labels, colors, max_lag=4000, var_names = [r'$u_R$', r'$u_I$', r'$v$']):
    """
    Plot time series, PDFs (log-scale), and ACFs.
    Layout: 3 rows (variables), 3 columns (time series | PDF | ACF)
    
    Parameters:
    - *_list: list of arrays of shape (T,)
    - sel0, sel1: time index range
    - interv: interval for time series downsampling
    - xlim: x-axis limits for time plots
    - ylim: x-axis limits for PDF plots (log x-scale)
    """
    from scipy.stats import gaussian_kde, norm
    from statsmodels.tsa.stattools import acf

    xaxis = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    lag_axis = dt * np.arange(max_lag + 1)

    fig = plt.figure(figsize=(10, 5))
    widths = [5, .8, .8]
    heights = [1, 1, 1]
    spec = fig.add_gridspec(nrows=3, ncols=3, width_ratios=widths, height_ratios=heights)
    plt.subplots_adjust(wspace=0.5, hspace=0.6)

    legend_lines = []
    legend_labels = []

    for row, (var_list, var_label) in enumerate(zip(data_groups, var_names)):
        ax_ts  = fig.add_subplot(spec[row, 0])  # Time series
        ax_pdf = fig.add_subplot(spec[row, 1])  # PDF
        ax_acf = fig.add_subplot(spec[row, 2])  # ACF

        for i, data in enumerate(var_list):
            samples = data[sel0:sel1]
            series = data[sel0:sel1:interv]

            # --- Time Series ---
            line_model, = ax_ts.plot(xaxis, series, color=colors[i], label=labels[i])
            if row == 0 and i == 0:  # only track lines from the first row to avoid duplicates
                legend_lines.append(line_model)
                legend_labels.append(labels[i])
        
            # --- PDF ---
            kde = gaussian_kde(samples)
            x_pdf = np.linspace(samples.min(), samples.max(), 300)
            p_pdf = kde.evaluate(x_pdf)
            ax_pdf.plot(p_pdf, x_pdf, color=colors[i])
            if i == 0:
                mean, std = samples.mean(), samples.std()
                gauss = norm.pdf(x_pdf, mean, std)
                line_gauss, = ax_pdf.plot(gauss, x_pdf, 'k--', label='Gaussian fit')
                if row == 0:
                    legend_lines.append(line_gauss)
                    legend_labels.append('Gaussian fit')
                
            # --- ACF ---
            acf_vals = acf(samples, nlags=max_lag, fft=True)
            ax_acf.plot(lag_axis, acf_vals, color=colors[i])

        # Axis labels/titles
        ax_ts.set_xlim(xlim)
        ax_ts.set_ylabel(var_label)
        ax_ts.set_xlabel("t")
        if row == 0:
            ax_ts.set_title("Time Series")

        # ax_pdf.set_xscale("log", base=10)
        if row == 0:
            ax_pdf.set_title("PDF")

        ax_acf.set_xlim(0, lag_axis[-1])
        # ax_acf.set_xlabel("lag (t)")
        if row == 0:
            ax_acf.set_title("ACF")

    # Global legend
    fig.legend(legend_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=len(legend_labels), fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
def plot_baro_series(dt, sel0, sel1, interv, xlim,
                           data_groups, v_field, T_field, labels, colors, var_names = [r'$U$', r'$\hat{v}_1$', r'$\hat{T}_1$']):
    """
    Plot time series, PDFs (log-scale), and ACFs.
    Layout: 3 rows (variables), 3 columns (time series | PDF | ACF)
    
    Parameters:
    - *_list: list of arrays of shape (T,)
    - sel0, sel1: time index range
    - interv: interval for time series downsampling
    - xlim: x-axis limits for time plots
    - ylim: x-axis limits for PDF plots (log x-scale)
    """

    xaxis = np.arange(sel0 * dt, sel1 * dt, interv * dt)

    fig = plt.figure(figsize=(10, 7))
    widths = [1]
    heights = [1, 1, 1, 1, 1]
    spec = fig.add_gridspec(nrows=5, ncols=1, width_ratios=widths, height_ratios=heights)
    plt.subplots_adjust(wspace=0.5, hspace=0.6)

    legend_lines = []
    legend_labels = []

    for row, (var_list, var_label) in enumerate(zip(data_groups, var_names)):
        ax_ts  = fig.add_subplot(spec[row, 0])  # Time series

        for i, data in enumerate(var_list):
            samples = data[sel0:sel1]
            series = data[sel0:sel1:interv]

            # --- Time Series ---
            line_model, = ax_ts.plot(xaxis, series, color=colors[i], label=labels[i], linewidth=.5)
            if row == 0 and i == 0:  # only track lines from the first row to avoid duplicates
                legend_lines.append(line_model)
                legend_labels.append(labels[i])
        
        # Axis labels/titles
        ax_ts.set_xlim(xlim)
        ax_ts.set_ylabel(var_label)
        # ax_ts.set_xlabel("t")
        if row == 0:
            ax_ts.set_title("Solutions of the Topographic Barotropic Model")


    # --- 4: v(x, t) field ---
    ax_vfield = fig.add_subplot(spec[3, 0])
    im1 = ax_vfield.imshow(
        v_field[sel0:sel1, :].T,
        origin='lower',
        aspect='auto',
        extent=[sel0 * dt, sel1 * dt, 0, np.pi*2],
        vmin=-10, vmax=10, 
        cmap='seismic'
    )
    ax_vfield.set_ylabel(r'$v$')
    # fig.colorbar(im1, ax=ax_vfield, shrink=0.9, pad=0.1, fraction=0.005)

    # --- 5: T(x, t) field ---
    ax_Tfield = fig.add_subplot(spec[4, 0])
    im2 = ax_Tfield.imshow(
        T_field[sel0:sel1, :].T,
        origin='lower',
        aspect='auto',
        extent=[sel0 * dt, sel1 * dt, 0, np.pi*2],
        vmin=-10, vmax=10, 
        cmap='seismic'
    )
    ax_Tfield.set_ylabel(r'$T$')
    ax_Tfield.set_xlabel("t")
    # fig.colorbar(im2, ax=ax_Tfield, shrink=0.9, pad=0.01, fraction=0.005)

    plt.tight_layout(rect=[0, 0.01, 1, 1])

    cbar = fig.colorbar(im2, ax=[ax_vfield, ax_Tfield], orientation='horizontal',
                        location='bottom', pad=0.22, fraction=0.02, aspect=30)
    # # Global legend
    # fig.legend(legend_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
    #            ncol=len(legend_labels), fontsize=8)
    # plt.tight_layout(rect=[0, 0, 1, 0.97])

def plot_baro_series_comparison(dt, sel0, sel1, interv, xlim,
                     data_groups, v_fields, T_fields, labels, colors,
                     var_names=[r'$U$', r'$\hat{v}_1$', r'$\hat{T}_1$'], line_width=1):
    """
    Plot time series and spatiotemporal fields v(x,t), T(x,t) for multiple methods.
    Layout: (3 time series) + (3 v fields) + (3 T fields)
    """
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    n_methods = len(v_fields)
    total_rows = 3 + 2 * n_methods  # 3 time series + n_methods v fields + n_methods T fields
    time_axis = np.arange(sel0 * dt, sel1 * dt, interv * dt)

    fig, axes = plt.subplots(total_rows, 1, figsize=(10, 1.3 * total_rows), sharex=False)
    plt.subplots_adjust(hspace=0.2)

    legend_lines = []
    legend_labels = []

    # --- Time series plots (top 3 rows) ---
    for i in range(3):
        ax = axes[i]
        for j, data in enumerate(data_groups[i]):
            ts = data[sel0:sel1:interv]
            line, = ax.plot(time_axis, ts, color=colors[j], linewidth=line_width)
            if i == 0:
                legend_lines.append(line)
                legend_labels.append(labels[j])
        ax.set_ylabel(var_names[i])
        ax.set_xlim(xlim)

    # --- v(x,t) fields ---
    for i in range(n_methods):
        ax = axes[3 + i]
        im_v = ax.imshow(
            v_fields[i][sel0:sel1].T,
            origin='lower',
            aspect='auto',
            extent=[sel0 * dt, sel1 * dt, 0, 2 * np.pi],
            vmin=-10, vmax=10,
            cmap='seismic'
        )
        ax.set_ylabel(r"$x$")
        # if i == 0:
        #     ax.annotate(r"$v(x,t)$", xy=(0.5, 1.02), xycoords='axes fraction',
        #                 fontsize=11, ha='center', va='bottom', fontweight='bold')

        ax.text(0.995, 0.95, labels[i],
                transform=ax.transAxes,
                fontsize=10,
                fontweight='bold',
                va='top', ha='right',
                bbox=dict(facecolor='white', alpha=0.2, edgecolor='none'))

    # --- T(x,t) fields ---
    for i in range(n_methods):
        ax = axes[3 + n_methods + i]
        im_T = ax.imshow(
            T_fields[i][sel0:sel1].T,
            origin='lower',
            aspect='auto',
            extent=[sel0 * dt, sel1 * dt, 0, 2 * np.pi],
            vmin=-10, vmax=10,
            cmap='seismic'
        )
        ax.set_ylabel(r"$x$")
        # if i == 0:
        #     ax.annotate(r"$T(x,t)$", xy=(0.5, 1.02), xycoords='axes fraction',
        #                 fontsize=11, ha='center', va='bottom', fontweight='bold')

        if i == n_methods - 1:
            ax.set_xlabel(r"$t$")
        ax.text(0.995, 0.95, labels[i],
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        va='top', ha='right',
        bbox=dict(facecolor='white', alpha=0.2, edgecolor='none'))

    # Add row labels using fig.text (global positioning)
    fig.text(0.5, 1 - (3.245) / total_rows, r"$v(x,t)$", fontsize=12, fontweight='bold', va='center', ha='left')
    fig.text(0.5, 1 - (5.754) / total_rows, r"$T(x,t)$", fontsize=12, fontweight='bold', va='center', ha='left')

    fig.suptitle("Free Forecast using LSTM model(s) at lead time $t=1$", fontsize=12, y=0.96)

    # --- Global legend (placed just below title) ---
    fig.legend(
        legend_lines, legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.945),  # slightly below the title
        ncol=len(labels),
        fontsize=9
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.96], pad=1.3)  # leave space for title and legend

    # --- Global colorbar (for T) ---
    cbar = fig.colorbar(im_T, ax=axes, location='bottom', pad=0.045, fraction=0.008, aspect=30)

def plot_pdf_and_joint(var1, var2, var_names, ylims, log=True):
    """
    Plot the log-scale PDF of var1 and var2 for given mode,
    and their joint PDF.

    Parameters:
    - var1: ndarray of shape (N,), real
    - var2: same shape as var1
    - mode: spectral mode k to plot (default is 1)
    - log: whether to use log-scale on x-axis (default True)
    """
    from scipy.stats import gaussian_kde, norm
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # --- 1. PDF of var1 ---
    kde_v = gaussian_kde(var1)
    x_v = np.linspace(var1.min(), var1.max(), 300)
    p_v = kde_v.evaluate(x_v)
    mean_v, std_v = var1.mean(), var1.std()
    gauss_v = norm.pdf(x_v, mean_v, std_v)

    axes[0].plot(x_v, p_v, label="Truth", color='k')
    axes[0].plot(x_v, gauss_v, 'k--', label="Gaussian fit")
    axes[0].set_title(f'PDF of {var_names[0]}')
    axes[0].legend()
    axes[0].set_ylim(ylims[0])
    if log:
        axes[0].set_yscale("log")

    # --- 2. PDF of var2 ---
    kde_T = gaussian_kde(var2)
    x_T = np.linspace(var2.min(), var2.max(), 300)
    p_T = kde_T.evaluate(x_T)
    mean_T, std_T = var2.mean(), var2.std()
    gauss_T = norm.pdf(x_T, mean_T, std_T)

    axes[1].plot(x_T, p_T, label="Truth", color='k')
    axes[1].plot(x_T, gauss_T, 'k--', label="Gaussian fit")
    axes[1].set_title(f'PDF of {var_names[1]}')
    axes[1].legend()
    axes[1].set_ylim(ylims[1])
    if log:
        axes[1].set_yscale("log")

    # --- 3. Joint PDF heatmap ---
    xy = np.vstack([var1, var2])
    kde_joint = gaussian_kde(xy)
    X, Y = np.meshgrid(
        np.linspace(var1.min(), var1.max(), 100),
        np.linspace(var2.min(), var2.max(), 100)
    )
    Z = kde_joint(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    im = axes[2].contourf(X, Y, Z, levels=50, cmap='binary')
    axes[2].set_title(f'Joint PDF of {var_names[0]} and {var_names[1]}')
    axes[2].set_xlabel(f'{var_names[0]}')
    axes[2].set_ylabel(f'{var_names[1]}')
    # fig.colorbar(im, ax=axes[2])

    # # --- 3. Joint PDF heatmap via histogram ---
    # ax = axes[2]
    # hist = ax.hist2d(
    #     re_vk, re_Tk,
    #     bins=50,
    #     density=True,       # normalize to get empirical joint PDF
    #     cmap='binary'
    # )
    # ax.set_title(f'Joint PDF of Re($\\hat{{v}}_{mode}$) and Re($\\hat{{T}}_{mode}$)')
    # ax.set_xlabel(f'Re($\\hat{{v}}_{mode}$)')
    # ax.set_ylabel(f'Re($\\hat{{T}}_{mode}$)')

    plt.tight_layout()
    

################################################### Clustering #####################################################    
def plot_scatter_weights(score, trueLabels, accuracy_entropy, accuracy_baseline, W_entropy, W_baseline,correct_entropy, correct_baseline, signalDim):
    # Plot entropy regularized results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Entropy Reg. (Acc: {accuracy_entropy*100:.1f}%)")
    plt.grid(True)
    plt.scatter(score[correct_entropy & (trueLabels == 1), 0], score[correct_entropy & (trueLabels == 1), 1], c='b', label='Class 1', s=50)
    plt.scatter(score[correct_entropy & (trueLabels == 2), 0], score[correct_entropy & (trueLabels == 2), 1], c='g', label='Class 2', s=50)
    plt.scatter(score[~correct_entropy, 0], score[~correct_entropy, 1], c='r', marker='x', label='Errors', s=50)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend()
    
    # Plot baseline results
    plt.subplot(1, 2, 2)
    plt.title(f"No Entropy (Acc: {accuracy_baseline*100:.1f}%)")
    plt.grid(True)
    plt.scatter(score[correct_baseline & (trueLabels == 1), 0], score[correct_baseline & (trueLabels == 1), 1], c='b', label='Class 1', s=50)
    plt.scatter(score[correct_baseline & (trueLabels == 2), 0], score[correct_baseline & (trueLabels == 2), 1], c='g', label='Class 2', s=50)
    plt.scatter(score[~correct_baseline, 0], score[~correct_baseline, 1], c='r', marker='x', label='Errors', s=50)
    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend()
    plt.tight_layout()
    
    # Plot feature weights
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(len(W_entropy)), W_entropy)
    plt.axvline(x=signalDim - 0.5, color='r', linestyle='--', linewidth=1.5)
    plt.title("Feature Weights with Entropy Regularization")
    plt.xlabel("Feature Index")
    plt.ylabel("Weight")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(len(W_baseline)), W_baseline)
    plt.axvline(x=signalDim - 0.5, color='r', linestyle='--', linewidth=1.5)
    plt.title("Feature Weights without Entropy")
    plt.xlabel("Feature Index")
    plt.ylabel("Weight")
    plt.grid(True)

    plt.tight_layout()

def plot_loss(loss_list):
    plt.figure(figsize=(4, 3))
    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # plt.title('Loss over Iterations')
    plt.grid(True)
    plt.tight_layout()

def plot_grad_norms(grad_norm_log):
    # Convert from list of PyTrees to arrays
    fuzzypart_norms = np.array([float(g[0]) for g in grad_norm_log])
    center_norms = np.array([float(g[1]) for g in grad_norm_log])
    W_norms = np.array([float(g[2]) for g in grad_norm_log])

    steps = np.arange(len(grad_norm_log))

    plt.figure(figsize=(6, 4))
    plt.plot(steps, fuzzypart_norms, label='fuzzypartmat logits', linewidth=2)
    plt.plot(steps, center_norms, label='centers', linewidth=2)
    plt.plot(steps, W_norms, label='W logits', linewidth=2)

    plt.yscale('log')  # log scale helps visualize small or diverging norms
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm (L2)')
    plt.title('Gradient Norms During Optimization')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()

def plot_l63_regimes(dt, sel0, sel1, interv, S,
                     prior_weights1, prior_weights2,
                     xlim, warmup=20):
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    fig, axes = plt.subplots(1, 1, figsize=(10, 2.3), sharex=True, gridspec_kw={'height_ratios': [1]})
    lines, labels = [], []

    # Regime plot
    ax = axes
    l1, = ax.plot(time, S[sel0:sel1:interv], 'k', linewidth=1.5, label='True Regime')
    l2, = ax.plot(time, prior_weights1[sel0:sel1:interv], 'b--', linewidth=1.5)
    l3, = ax.plot(time, prior_weights2[sel0:sel1:interv], 'r--', linewidth=1.5)
    
    # Legend setup
    lines.extend([l1, l2, l3])
    labels.extend([
        'True Regime',
        'Membership w/o feature selection',
        'Membership w/ feature selection'
    ])

    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel('$t$')
    ax.set_title('FCM clustering with / without entropy-regularized feature selection', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(xlim)

    # Global legend
    fig.legend(
        handles=lines,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.51, 0.12),
        ncol=3,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])  # leave enough bottom margin


###################################################### DA #######################################################    
def plot_l63_series(dt, sel0, sel1, interv,
                    x_truth, y_truth, z_truth, S,
                    mean, spread,
                    prior_weights, posterior_weights,
                    xlim, warmup=20, prior_mean=None, prior_spread=None, obs=None):
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    var_names = ['X', 'Y', 'Z']
    truth_vars = [x_truth, y_truth, z_truth]
    std = np.sqrt(spread)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    lines, labels = [], []

    # Time series plots (X, Y, Z)
    for i in range(3):
        ax = axes[i]
        l1, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=1.5, label='Truth')
        l2, = ax.plot(time, mean[sel0:sel1:interv, i], 'r', linewidth=1.5, label='Posterior Mean')
        l3 = ax.fill_between(time,
                             mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
                             mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
                             color='r', alpha=0.2, label='Spread')
        if prior_mean is not None:
            prior_std = np.sqrt(prior_spread)
            ax.plot(time, prior_mean[sel0:sel1:interv, i], 'b', linewidth=1.5, label='Prior Mean')
            ax.fill_between(time,
                             prior_mean[sel0:sel1:interv, i] - prior_std[sel0:sel1:interv, i],
                             prior_mean[sel0:sel1:interv, i] + prior_std[sel0:sel1:interv, i],
                             color='b', alpha=0.2, label='Prior Spread')
        if obs is not None:
            ax.plot(time, obs[sel0:sel1:interv, i], 'g', linewidth=1.5, label='Obs')

        ax.set_title(var_names[i], fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlim(xlim)

        if i == 0:
            lines.extend([l1, l2, l3])
            labels.extend(['Truth', 'Posterior Mean', 'Posterior Spread'])

        # Correlation and RMSE annotation
        truth_i = truth_vars[i][warmup:]
        mean_i = mean[warmup:, i]
        corr = np.corrcoef(truth_i, mean_i)[0, 1]
        rmse = np.sqrt(np.mean((truth_i - mean_i) ** 2)) # time mean rmse
        textstr = f'Corr = {corr:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.99, 0.94, textstr,
                transform=ax.transAxes,
                fontsize=9,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Regime plot
    ax = axes[3]
    l4, = ax.plot(time, S[sel0:sel1:interv], 'k', linewidth=1.5, label='True Regime')
    l5, = ax.plot(time, prior_weights[sel0:sel1:interv], 'b--', linewidth=1.5, label='Prior Weight')
    l6, = ax.plot(time, posterior_weights[sel0:sel1:interv], 'r--', linewidth=1.5, label='Posterior Weight')
    lines.extend([l4, l5, l6])
    labels.extend(['True Regime', 'Prior Weight', 'Posterior Weight'])
    ax.set_ylim([-0.1, 1.1])
    ax.set_title('Regime', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(xlim)

    # Global legend
    fig.legend(
        handles=lines,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.51, 0.04),
        ncol=6,
        fontsize=10,
    )

    # fig.tight_layout()  # Leave space for the global legend    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave enough bottom margin

def plot_ou_series(dt, sel0, sel1, interv,
                    ur_truth, ui_truth, v_truth, 
                    mean=None, spread=None,
                    prior_weights=None, posterior_weights=None,
                    xlim=None, warmup=20, prior_mean=None, prior_spread=None, obs=None, S=None, var_names=['uR', 'uI', 'v']):
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    truth_vars = [ur_truth, ui_truth, v_truth]
    std = np.sqrt(spread)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    lines, labels = [], []

    # Time series plots 
    for i in range(3):
        ax = axes[i]
        l1, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=1.5, label='Truth')
        l2, = ax.plot(time, mean[sel0:sel1:interv, i], 'r', linewidth=1.5, label='Posterior Mean')
        l3 = ax.fill_between(time,
                             mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
                             mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
                             color='r', alpha=0.2, label='Spread')
        if prior_mean is not None:
            prior_std = np.sqrt(prior_spread)
            ax.plot(time, prior_mean[sel0:sel1:interv, i], 'b', linewidth=1.5, label='Prior Mean')
            ax.fill_between(time,
                             prior_mean[sel0:sel1:interv, i] - prior_std[sel0:sel1:interv, i],
                             prior_mean[sel0:sel1:interv, i] + prior_std[sel0:sel1:interv, i],
                             color='b', alpha=0.2, label='Prior Spread')
        if obs is not None:
            ax.plot(time, obs[sel0:sel1:interv, i], 'g', linewidth=1.5, label='Obs')

        ax.set_title(var_names[i], fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlim(xlim)

        if i == 0:
            lines.extend([l1, l2, l3])
            labels.extend(['Truth', 'Posterior Mean', 'Posterior Spread'])

        if i < 2:
            # Correlation and RMSE annotation
            truth_i = truth_vars[i][warmup:]
            mean_i = mean[warmup:, i]
            corr = np.corrcoef(truth_i, mean_i)[0, 1]
            rmse = np.sqrt(np.mean((truth_i - mean_i) ** 2)) # time mean rmse
            textstr = f'Corr = {corr:.3f}\nRMSE = {rmse:.3f}'
            ax.text(0.99, 0.94, textstr,
                    transform=ax.transAxes,
                    fontsize=9,
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Regime plot
    ax = axes[3]

    if S is not None:
        l4, = ax.plot(time, S[sel0:sel1:interv], 'k', linewidth=1.5, label='True Regime')
        lines.extend([l4])
        labels.extend(['True Regime'])
    if prior_weights is not None:
        l5, = ax.plot(time, prior_weights[sel0:sel1:interv], 'b--', linewidth=1.5, label='Prior Weight')
        lines.extend([l5])
        labels.extend(['Prior Weight'])
    if posterior_weights is not None:
        l6, = ax.plot(time, posterior_weights[sel0:sel1:interv], 'r--', linewidth=1.5, label='Posterior Weight')
        lines.extend([l6])
        labels.extend(['Posterior Weight'])
    ax.set_ylim([-0.1, 1.1])
    ax.set_title('Regime', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(xlim)

    # Global legend
    fig.legend(
        handles=lines,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.51, 0.04),
        ncol=6,
        fontsize=10,
    )

    # fig.tight_layout()  # Leave space for the global legend    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave enough bottom margin

def plot_ou_series_comparison(dt, sel0, sel1, interv, truth_vars, 
                    mean=None, spread=None,
                    prior_weights=None, posterior_weights=None,
                    xlim=None, warmup=20, prior_mean=None, prior_spread=None, obs=None, S=None, 
                    var_names = ['Single-model EnKF', 'Standard Multi-model EnKF', 'Stochastic Parameterization EnKF', 'Gaussian Mixture Multi-model EnKF', 'Stochastic Parameterization EnKF: $v$', 'Gaussian Mixture Multi-model EnKF: regime'],
                    y_labels = ['Re[$u$]', 'Re[$u$]', 'Re[$u$]', 'Re[$u$]', '$v$', 'weight']):
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    std = np.sqrt(spread)

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1]})
    lines, labels = [], []

    # Time series plots 
    for i in range(len(truth_vars)):
        ax = axes[i]
        l1, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=1.5, label='Truth')
        l2, = ax.plot(time, mean[sel0:sel1:interv, i], 'r', linewidth=1.5, label='Posterior Mean')
        l3 = ax.fill_between(time,
                             mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
                             mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
                             color='r', alpha=0.2, label='Spread')
        ax.set_ylabel(y_labels[i])
        if prior_mean is not None:
            prior_std = np.sqrt(prior_spread)
            ax.plot(time, prior_mean[sel0:sel1:interv, i], 'b', linewidth=1.5, label='Prior Mean')
            ax.fill_between(time,
                             prior_mean[sel0:sel1:interv, i] - prior_std[sel0:sel1:interv, i],
                             prior_mean[sel0:sel1:interv, i] + prior_std[sel0:sel1:interv, i],
                             color='b', alpha=0.2, label='Prior Spread')
        if obs is not None:
            ax.plot(time, obs[sel0:sel1:interv, i], 'g', linewidth=1.5, label='Obs')

        ax.set_title(var_names[i], fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlim(xlim)

        if i == 0:
            lines.extend([l1, l2, l3])
            labels.extend(['Truth', 'Posterior Mean', 'Posterior Spread'])

        if i < len(truth_vars)-1:
            # Correlation and RMSE annotation
            truth_i = truth_vars[i][warmup:]
            mean_i = mean[warmup:, i]
            corr = np.corrcoef(truth_i, mean_i)[0, 1]
            rmse = np.mean(np.sqrt((truth_i - mean_i) ** 2)) # time mean rmse
            textstr = f'Corr = {corr:.3f}\nRMSE = {rmse:.3f}'
            ax.text(0.99, 0.94, textstr,
                    transform=ax.transAxes,
                    fontsize=9,
                    fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Regime plot
    ax = axes[-1]

    if S is not None:
        l4, = ax.plot(time, S[sel0:sel1:interv], 'k', linewidth=1.5, label='True Regime')
        lines.extend([l4])
        labels.extend(['True Regime'])
    if prior_weights is not None:
        l5, = ax.plot(time, prior_weights[sel0:sel1:interv], 'b--', linewidth=1.5, label='Prior Weight')
        ax.set_ylabel('weight')
        lines.extend([l5])
        labels.extend(['Prior Weight'])
    if posterior_weights is not None:
        l6, = ax.plot(time, posterior_weights[sel0:sel1:interv], 'r--', linewidth=1.5, label='Posterior Weight')
        lines.extend([l6])
        labels.extend(['Posterior Weight'])
    ax.set_ylim([-0.1, 1.1])
    ax.set_title(var_names[-1], fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(xlim)

    # Global legend
    fig.legend(
        handles=lines,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.51, 0.04),
        ncol=6,
        fontsize=10,
    )

    # fig.tight_layout()  # Leave space for the global legend    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave enough bottom margin

def plot_topobaro_series(dt, sel0, sel1, interv,
                    truth_vars, 
                    mean=None, spread=None,
                    prior_weights=None, posterior_weights=None,
                    xlim=None, warmup=20, prior_mean=None, prior_spread=None, obs=None, S=None, var_names=['$U$', '$v_1$', '$T_1$']):
    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    std = np.sqrt(spread)

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    lines, labels = [], []

    # Time series plots 
    for i in range(3):
        ax = axes[i]
        l1, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=1.5, label='Truth')
        l2, = ax.plot(time, mean[sel0:sel1:interv, i], 'r', linewidth=1.5, label='Posterior Mean')
        l3 = ax.fill_between(time,
                             mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
                             mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
                             color='r', alpha=0.2, label='Spread')
        if prior_mean is not None:
            prior_std = np.sqrt(prior_spread)
            ax.plot(time, prior_mean[sel0:sel1:interv, i], 'b', linewidth=1.5, label='Prior Mean')
            ax.fill_between(time,
                             prior_mean[sel0:sel1:interv, i] - prior_std[sel0:sel1:interv, i],
                             prior_mean[sel0:sel1:interv, i] + prior_std[sel0:sel1:interv, i],
                             color='b', alpha=0.2, label='Prior Spread')
        if obs is not None:
            ax.plot(time, obs[sel0:sel1:interv, i], 'g', linewidth=1.5, label='Obs')

        ax.set_title(var_names[i], fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlim(xlim)

        if i == 0:
            lines.extend([l1, l2, l3])
            labels.extend(['Truth', 'Posterior Mean', 'Posterior Spread'])

        # if i < 2:
        # Correlation and RMSE annotation
        truth_i = truth_vars[i][warmup:]
        mean_i = mean[warmup:, i]
        corr = np.corrcoef(truth_i, mean_i)[0, 1]
        rmse = np.sqrt(np.mean((truth_i - mean_i) ** 2)) # time mean rmse
        textstr = f'Corr = {corr:.3f}\nRMSE = {rmse:.3f}'
        ax.text(0.99, 0.94, textstr,
                transform=ax.transAxes,
                fontsize=9,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Regime plot
    ax = axes[3]

    if S is not None:
        l4, = ax.plot(time, S[sel0:sel1:interv], 'k', linewidth=1.5, label='True Regime')
        lines.extend([l4])
        labels.extend(['True Regime'])
    if prior_weights is not None:
        l5, = ax.plot(time, prior_weights[sel0:sel1:interv], 'b--', linewidth=1.5, label='Prior Weight')
        lines.extend([l5])
        labels.extend(['Prior Weight'])
    if posterior_weights is not None:
        l6, = ax.plot(time, posterior_weights[sel0:sel1:interv], 'r--', linewidth=1.5, label='Posterior Weight')
        lines.extend([l6])
        labels.extend(['Posterior Weight'])
    ax.set_ylim([-0.1, 1.1])
    ax.set_title('Regime', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(xlim)

    # Global legend
    fig.legend(
        handles=lines,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.51, 0.04),
        ncol=6,
        fontsize=10,
    )

    # fig.tight_layout()  # Leave space for the global legend    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave enough bottom margin

def plot_topobaro_series_comparison(dt, sel0, sel1, interv,
                    truth_vars, means=None, spreads=None,
                    prior_weights=None, posterior_weights=None,
                    xlim=None, warmup=40, S=None, var_names=['$U$', '$v_1$', '$T_1$'],
                    mean_labels=None, colors=['r', 'b', 'g', 'orange', 'purple', 'brown'], line_width=1.5, title=None):

    time = np.arange(sel0 * dt, sel1 * dt, interv * dt)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
    lines, labels = [], []
    if title is  not None:
        axes[0].set_title(title, fontsize=14)
    
    # Time series plots 
    for i in range(3):
        ax = axes[i]
        l_truth, = ax.plot(time, truth_vars[i][sel0:sel1:interv], 'k', linewidth=line_width)
        ax.set_ylabel(var_names[i], fontsize=12)
        ax.tick_params(labelsize=10)
        ax.set_xlim(xlim)
        if i == 0:
            lines.append(l_truth)
            labels.append('Truth')

        for mean, label, color in zip(means, mean_labels, colors):
            l_mean, = ax.plot(time, mean[sel0:sel1:interv, i], color=color, linewidth=line_width)
            if i == 0:
                lines.append(l_mean)
                labels.append(label)
        if spreads is not None:
            for std, mean, color in zip(spreads, means, colors):
                l_spread = ax.fill_between(time,
                             mean[sel0:sel1:interv, i] - std[sel0:sel1:interv, i],
                             mean[sel0:sel1:interv, i] + std[sel0:sel1:interv, i],
                             color=color, alpha=0.2)
        
        # Prepare correlation and RMSE in a transposed layout
        corrs, rmses = [], []
        for mean in means:
            truth_i = truth_vars[i][warmup:]
            mean_i = mean[warmup:, i]
            corr = np.corrcoef(truth_i, mean_i)[0, 1]
            rmse = np.sqrt(np.mean((truth_i - mean_i) ** 2))
            corrs.append(f'{corr:.3f}')
            rmses.append(f'{rmse:.3f}')

        # Build two rows: one for correlation, one for RMSE
        corr_row = r"Corr:  " + "   ".join([f"{c}" for c in corrs])
        rmse_row = r"RMSE: " + "   ".join([f"{r}" for r in rmses])
        metrics_box_text = f"{corr_row}\n{rmse_row}"

        # Draw a transparent dummy text to get the box
        ax.text(
            0.99, 0.94,
            metrics_box_text,
            transform=ax.transAxes,
            fontsize=9,
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='right',
            color=(0, 0, 0, 0),  # fully transparent text
            # color='black',  # fully transparent text
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

        # Overlay visible text in each cell
        line_spacing_x = 0.058
        line_spacing_y = 0.1
        ax.text(
            0.82, 0.94,  # adjust x-spacing
            'Corr:',
            transform=ax.transAxes,
            fontsize=9,
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='right',
            color='black'
        )
        ax.text(
            0.83, 0.94 - line_spacing_y,
            'RMSE:',
            transform=ax.transAxes,
            fontsize=9,
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='right',
            color='black'
        )
        for j, color in enumerate(colors[:len(means)]):
            ax.text(
                1.05 - (len(means)-j) * line_spacing_x, 0.94,  # adjust x-spacing
                corrs[j],
                transform=ax.transAxes,
                fontsize=9,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right',
                color=color
            )
            ax.text(
                1.05 - (len(means)-j) * line_spacing_x, 0.94 - line_spacing_y,
                rmses[j],
                transform=ax.transAxes,
                fontsize=9,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='right',
                color=color
            )

    # Regime plot
    ax = axes[3]
    if S is not None:
        l4, = ax.plot(time, S[sel0:sel1:interv], 'k', linewidth=line_width)
        lines.append(l4)
        labels.append('True Regime')
    if prior_weights is not None:
        l5, = ax.plot(time, prior_weights[sel0:sel1:interv], 'b--', linewidth=line_width)
        lines.append(l5)
        labels.append('Prior Weight')
    if posterior_weights is not None:
        l6, = ax.plot(time, posterior_weights[sel0:sel1:interv], 'r--', linewidth=line_width)
        lines.append(l6)
        labels.append('Posterior Weight')

    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel('weight', fontsize=12)
    ax.set_xlabel('t', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlim(xlim)

    # Global legend
    fig.legend(
        handles=lines,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.51, 0.04),
        ncol=6,
        fontsize=10,
    )
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

def plot_topobaro_fields_comparison(dt, sel0, sel1, v_fields, T_fields, method_labels, vlim=10):
    """
    Plot v(x, t) and T(x, t) fields for multiple methods.
    """
    n_methods = len(method_labels)
    time_extent = [sel0 * dt, sel1 * dt]
    x_extent = [0, 2 * np.pi]
    fig, axes = plt.subplots(n_methods, 2, figsize=(10, 1.2 * n_methods), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    for i in range(n_methods):
        ax_v = axes[i, 0]
        im_v = ax_v.imshow(v_fields[i][sel0:sel1].T,
                           origin='lower',
                           aspect='auto',
                           extent=time_extent + x_extent,
                           vmin=-vlim, vmax=vlim,
                           cmap='seismic')
        ax_v.set_ylabel(r"$x$", fontsize=10)

        ax_T = axes[i, 1]
        im_T = ax_T.imshow(T_fields[i][sel0:sel1].T,
                           origin='lower',
                           aspect='auto',
                           extent=time_extent + x_extent,
                           vmin=-vlim, vmax=vlim,
                           cmap='seismic')
        if i == n_methods - 1:
            ax_v.set_xlabel(r"$t$")
            ax_T.set_xlabel(r"$t$")
        ax_T.text(0.99, 0.95, method_labels[i],
          transform=ax_T.transAxes,
          fontsize=10,
          va='top',
          ha='right',
          bbox=dict(facecolor='white', alpha=0.1, edgecolor='none'))
    axes[0, 0].set_title(r"$v(x,t)$", fontsize=11)
    axes[0, 1].set_title(r"$T(x,t)$", fontsize=11)
    plt.tight_layout(rect=[0, 0.01, 1, 1])
    cbar = fig.colorbar(im_T, ax=axes, location='bottom', pad=0.1, fraction=0.015, aspect=40)

def plot_gmm_pdf_3vars(means_list, stds_list, weights_list, x_ranges=None, num_points=1000, var_names=['X', 'Y', 'Z']):
    """
    Plot the PDFs of 1D Gaussian mixtures for three variables in subplots.

    Parameters:
    - means_list: list of 3 arrays of means for each variable
    - stds_list: list of 3 arrays of stds for each variable
    - weights_list: list of 3 arrays of weights for each variable
    - x_ranges: optional list of 3 (xmin, xmax) tuples; if None, auto-determined
    - num_points: number of x-points per PDF
    - var_names: list of variable names for labeling
    """
    from scipy.stats import norm
    fig, axes = plt.subplots(1, 3, figsize=(10,3), sharex=False)

    for i in range(3):
        means = np.asarray(means_list[i])
        stds = np.asarray(stds_list[i])
        weights = np.asarray(weights_list)

        if x_ranges is None or x_ranges[i] is None:
            xmin = np.min(means - 4 * stds)
            xmax = np.max(means + 4 * stds)
        else:
            xmin, xmax = x_ranges[i]

        x = np.linspace(xmin, xmax, num_points)
        pdf = np.zeros_like(x)
        for mu, sigma, w in zip(means, stds, weights):
            pdf += w * norm.pdf(x, loc=mu, scale=sigma)

        ax = axes[i]
        ax.plot(x, pdf, 'k')
        for j, mu in enumerate(means):
            ax.axvline(mu, linestyle='--', color='k', alpha=0.5, label=f'$\mu_{j}={mu:.2f}$')
        ax.set_xlabel(f"{var_names[i]}")
        # ax.set_title(f"GMM PDF of {var_names[i]}")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Density")
    plt.tight_layout()

def plot_gmm_pdf_3vars_multi(
    means_array, stds_array, weights_array,
    x_ranges=None, num_points=1000, var_names=['X', 'Y', 'Z'],
    labels=None, colors=None, title=None, truth_array=None
):
    """
    Plot PDFs of 1D Gaussian mixtures for three variables, each with multiple GMMs.

    Parameters:
    - means_array: (K, 3, N) array of means
    - stds_array: (K, 3, N) array of stds
    - weights_array: (K, 3, N) array of weights
    - x_ranges: optional list of 3 (xmin, xmax) tuples
    - num_points: number of x-points per PDF
    - var_names: list of variable names for labeling
    - labels: list of K strings for legend labels
    - colors: list of K colors for different GMM curves
    - truth_array: 
    """
    from scipy.stats import norm

    K = means_array.shape[0]  # number of GMMs
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=False)
    legend_lines = []
    if labels is None:
        labels = [f"GMM {i}" for i in range(K)]
    if colors is None:
        colors = ['k', 'r', 'b'][:K]

    for i in range(3):  # for each variable
        ax = axes[i]
        for k in range(K):  # for each GMM
            means = means_array[k, i]
            stds = stds_array[k, i]
            weights = weights_array[k, i]

            # Trim unused components if weights are padded with zeros
            mask = weights > 0
            means = means[mask]
            stds = stds[mask]
            weights = weights[mask]

            if x_ranges is None or x_ranges[i] is None:
                xmin = np.min(means - 4 * stds)
                xmax = np.max(means + 4 * stds)
            else:
                xmin, xmax = x_ranges[i]

            x = np.linspace(xmin, xmax, num_points)
            pdf = np.zeros_like(x)
            for mu, sigma, w in zip(means, stds, weights):
                pdf += w * norm.pdf(x, loc=mu, scale=sigma)

            line, = ax.plot(x, pdf, color=colors[k], label=labels[k], linewidth=2)
            for mu in means:
                ax.axvline(mu, linestyle='--', color=colors[k], alpha=0.3)
            if i == 0:  # only collect once
                legend_lines.append(line)
                        
        line = ax.axvline(truth_array[i], linestyle='--', color=colors[K], alpha=0.8)
        if i==0:
            legend_lines.append(line)
        ax.set_xlabel(f"{var_names[i]}")

    axes[0].set_ylabel("Density")
    fig.suptitle(title, fontsize=14)
    fig.legend(
        handles=legend_lines,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.12),  # adjust for layout
        ncol=K+1,
        fontsize=10
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])  # leave space for legend
    

######################################### Model Evaluation ###########################################    
def plot_all_histograms_univar(hist_data, variables=['x', 'y', 'z'], figsize=(12, 4)):
    n_regimes = len(hist_data[variables[0]])
    n_vars = len(variables)

    fig, axes = plt.subplots(n_vars, n_regimes, figsize=figsize, squeeze=False)

    for i in range(n_regimes):
        for j, var in enumerate(variables):
            p_hat, q_hat, bin_edges, regime_id, model_id = hist_data[var][i]
            ax = axes[j][i]
            bin_edges = bin_edges[0]
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            width = (bin_edges[1] - bin_edges[0]) * 0.4

            ax.bar(bin_centers - width/2, p_hat, width=width, color='k', label='Truth', alpha=0.7)
            ax.bar(bin_centers + width/2, q_hat, width=width, color='r', label='Model', alpha=0.7)

            ax.set_title(f'Regime {regime_id}, Model {model_id}', fontsize=10)
            if i == 0:
                ax.set_ylabel(var)

            if i == 0 and j == 0:
                ax.legend(fontsize=6.5)

    fig.tight_layout()
    return fig

def plot_histogram_comparison(p_hat, q_hat, bin_edges, title='', var_name='x'):
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = (bin_edges[1] - bin_edges[0]) * 0.4

    plt.figure(figsize=(6, 4))
    plt.bar(bin_centers - width/2, p_hat, width=width, color='k', label='Truth', alpha=0.7)
    plt.bar(bin_centers + width/2, q_hat, width=width, color='r', label='Model', alpha=0.7)
    
    plt.xlabel(f'{var_name}')
    plt.ylabel('Probability')
    plt.title(title or f'Histogram comparison for {var_name}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

def plot_3d_histogram(p, edges, title='3D Histogram', threshold=0.001):
    """
    Plot a 3D voxel representation of a 3D joint histogram using bin edges.
    
    Parameters:
    - p: 3D numpy array of histogram probabilities
    - edges: list of 3 arrays for bin edges in x, y, z
    - title: plot title
    - threshold: minimum density to display
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Only show voxels above threshold
    filled = p > threshold
    values = p[filled] / p.max()

    # Get bin centers
    x_centers = 0.5 * (edges[0][:-1] + edges[0][1:])
    y_centers = 0.5 * (edges[1][:-1] + edges[1][1:])
    z_centers = 0.5 * (edges[2][:-1] + edges[2][1:])
    
    # Build 3D grid of centers
    X, Y, Z = np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')

    # Extract coordinates where filled
    x_vals = X[filled]
    y_vals = Y[filled]
    z_vals = Z[filled]

    # Prepare color array
    facecolors = plt.cm.gist_gray(values)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # Plot each voxel manually as a cube
    size = (edges[0][1] - edges[0][0])  # assume equal-sized bins
    for (x, y, z, color) in zip(x_vals, y_vals, z_vals, facecolors):
        ax.bar3d(x, y, z, size, size, size, color=color, edgecolor='k', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.tight_layout()

######################################### Schematic ###########################################    
def plot_vertical_gaussian_mixture(components= [
                                    (0.1, -.6, 0.4),
                                    (0.3, 1.2,1.2),
                                    (0.6, 1.6, 0.5)], x_range=(-5, 5), resolution=1000,
                                    highlight_idx=None, figsize=(2, 2), color='r'):
    """
    Plot a vertical Gaussian mixture model (PDF on x-axis, values on y-axis).

    Args:
        components: List of (weight, mean, std) tuples.
        x_range: Tuple (xmin, xmax) for the range of x values.
        resolution: Number of x points to sample.
        highlight_idx: Index of a specific component to plot instead of the full mixture.
        figsize: Size of the plot.
        color: Color for the mixture or component curve.
    """
    x = np.linspace(*x_range, resolution)
    y_components = []
    y_mixture = np.zeros_like(x)

    for weight, mu, sigma in components:
        y = weight * norm.pdf(x, mu, sigma)
        y_components.append(y)
        y_mixture += y

    plt.figure(figsize=figsize)

    if highlight_idx is not None:
        plt.plot(y_components[highlight_idx], x, color, linewidth=4.5,
                 label=f'Component {highlight_idx}')
    else:
        plt.plot(y_mixture, x, color, linewidth=4.5, label='Gaussian Mixture')

    plt.ylim(*x_range)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

######################################### ENSO ########################################### 
def plot_eofs(eofs, evr, modes=(1,2,3), vmin=None, vmax=None):
    modes = list(modes)
    fig, axs = plt.subplots(len(modes), 1, figsize=(6, 2*len(modes)), constrained_layout=True)
    if len(modes) == 1: axs = [axs]
    for ax, m in zip(axs, modes):
        patt = eofs.sel(mode=m)
        patt = patt.sortby("lat")
        im = patt.plot.imshow(
            ax=ax, cmap="RdBu_r", vmin=vmin, vmax=vmax,
            add_colorbar=False
        )
        ax.set_title("")
        ax.text(
            0.02, 0.02, f"EOF {m} ({evr.sel(mode=m).item()*100:.1f}% var)",
            transform=ax.transAxes, fontsize=10,
            color="black", ha="left", va="bottom",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2)
        )
        if ax != axs[-1]:
            ax.set_xlabel("")
        else: 
            ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cbar = fig.colorbar(im, ax=axs, orientation="vertical", aspect=50, shrink=0.8, fraction=0.1, pad=0.02)
    fig.suptitle("EOFs", fontsize=12)

def plot_pcs(pcs, modes=(1,2,3)):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    for m in modes:
        ax.plot(pcs["time"].values, pcs.sel(mode=m), label=f"PC{m}")
    ax.legend()
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("Time")
    ax.set_ylabel("PC (arb. units)")
    plt.tight_layout()
    plt.show()

def plot_nino34_with_regimes(
    nino34,                   # np.ndarray, shape (Nt,)
    labels,                   # np.ndarray or list, shape (Nt,), ints like 0..K-1 (or -1 for noise)
    time=None,                # None, np.ndarray shape (Nt,), or pandas.DatetimeIndex
    regime_names=None,        # list of names per regime id; len==K
    colors=None,              # list of color strings per regime id; len==K
    rolling=None,             # e.g., 3 or 5 for moving avg window (optional)
    title="Niño 3.4 with clustered regimes"
):
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap
    nino34 = np.asarray(nino34)
    labels = np.asarray(labels)
    assert nino34.shape == labels.shape, "nino34 and labels must have same shape"
    Nt = nino34.size
    if time is None:
        x = np.arange(Nt)
        xlabel = "Time index"
    else:
        x = np.asarray(time)
        xlabel = "Time"
        
    # Handle NaNs: mask both arrays consistently
    good = np.isfinite(nino34) & np.isfinite(labels)
    x_plot = x[good]
    y_plot = nino34[good]
    lab_plot = labels[good].astype(int)

    # Unique regimes (keep order by sorted)
    uniq = np.unique(lab_plot)

    # Build colors
    assert len(colors) >= len(uniq), "Provide enough colors for all regimes"
    palette = colors[:len(uniq)]

    # Map regime id -> color (keep label ids)
    color_map = {lab: palette[i] for i, lab in enumerate(uniq)}

    # Build names
    name_map = {lab: (regime_names[lab] if lab >= 0 else "Noise") for lab in uniq} # If regime ids start at 0..K-1, map directly; for -1 keep "Noise"

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x, nino34, 'k', lw=0.9, alpha=0.6)
    for lab in uniq:
        sel = lab_plot == lab
        ax.scatter(x_plot[sel], y_plot[sel], s=14, color=color_map[lab], label=name_map[lab], zorder=3)
    ax.set_ylabel("SST anomaly (°C)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Deduplicate legend entries (Niño line, rolling mean, regimes)
    handles, labels_ = ax.get_legend_handles_labels()
    seen = {}
    new_h, new_l = [], []
    for h, l in zip(handles, labels_):
        if l not in seen:
            seen[l] = True
            new_h.append(h); new_l.append(l)
    ax.legend(new_h, new_l, ncol=2, fontsize=9)
    # Legend outside bottom
    ax.legend(
        new_h, new_l,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(new_l),
        fontsize=9
    )

    plt.tight_layout()
    return fig, ax

def plot_five_regime_means(mean_maps: xr.DataArray, freq: xr.DataArray, regimes=None, cmap="RdBu_r"):
    """
    Plot five regime-mean anomaly maps with a shared symmetric color scale.
    If 'regimes' is None, the first five regimes in mean_maps.regime are used.
    """
    if regimes is None:
        regimes = mean_maps.regime.values[:5]
    sel = mean_maps.sel(regime=regimes)

    vmax = float(np.nanmax(np.abs(sel.values)))
    vmin = -vmax

    fig, axs = plt.subplots(
        len(regimes), 1, figsize=(6, 1.5*len(regimes)),
        sharex=True, constrained_layout=True
    )

    if len(regimes) == 1:
        axs = [axs]
        
    frq_map = {int(r): float(freq.sel(regime=r).item()) for r in freq.regime.values}

    for ax, reg in zip(axs, regimes):
        patt = sel.sel(regime=reg)
        im = patt.plot.imshow(
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False
        )
        ax.set_ylabel("Latitude")
        if ax != axs[-1]:
            ax.set_xlabel("")
        else: 
            ax.set_xlabel("Longitude")
        ax.set_title("")  # remove any auto-generated title
        r = int(reg)
        frq = frq_map.get(r, 0.0) * 100.0
        ax.text(
            0.02, 0.02, f"Regime {reg} ({frq:.1f}%)",
            transform=ax.transAxes, fontsize=10,
            color="black", ha="left", va="bottom",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=2)
        )

    cbar = fig.colorbar(im, ax=axs, orientation="vertical", aspect=50, shrink=0.8, fraction=0.1, pad=0.02)
    fig.suptitle("Mean SSTA by regime", fontsize=12)



