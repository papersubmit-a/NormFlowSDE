import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.colors as mcolors
from sklearn.mixture import GaussianMixture


def has_content(ax):
    return any([
        ax.lines,               # line plots
        ax.patches,             # bars, filled areas
        ax.images,              # imshow, heatmap
        ax.collections,         # scatter, contourf
        ax.texts[1:],           # user text (ignore title)
    ])

def plot_single_parameter_comparison(ax, gt_i, mean_i, ci_i, ci_color, gt_color, main_color, confidence_level, param_name_i):
    # Determine plot limits based on min/max of both GT and mean estimates
    min_val = min(gt_i.min(), mean_i.min())
    max_val = max(gt_i.max(), mean_i.max())
    plot_range = max_val - min_val
    padding = plot_range * 0.1
    plot_min = min_val - padding
    plot_max = max_val + padding

    # --- Scatter of All Estimates (Optional, can be messy, often omitted for clarity) ---
    # Instead, we focus on mean and CI. If we plot all, the shape is (N_samples * N_estimates)
    # We can flatten the estimates for a general scatter plot:
    # flat_estimates = estimated_params[:, :, i].flatten()
    # flat_gt = np.repeat(gt_i, estimated_params.shape[1])
    # ax.scatter(flat_gt, flat_estimates, color='grey', alpha=0.1, label='All Estimates')

    # --- Confidence Intervals (Error Bars) ---
    ax.errorbar(
        gt_i,
        mean_i,
        yerr=ci_i,
        fmt='o',
        color=ci_color,
        alpha=0.6,
        capsize=3,
        label=fr'Mean $\pm$ {confidence_level * 100:.0f}% CI'
    )

    # --- Mean Comparison (Scatter) ---
    # Plot the mean values (on top of the error bars for better visibility)
    ax.scatter(gt_i, mean_i, color=main_color, marker='o', s=30, zorder=3, label='Mean Estimates')

    # --- Ground Truth Line (Y=X line) ---
    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        '--',
        color=gt_color,
        alpha=0.7,
        label='Ground Truth (Y=X)'
    )

    # --- Set Axes and Labels ---
    ax.set_title(f'Parameter Regression: {param_name_i}')
    ax.set_xlabel(f'Ground Truth {param_name_i}')
    ax.set_ylabel(f'Estimated {param_name_i}')
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect('equal', adjustable='box')  # Enforce square aspect ratio
    ax.grid(True, linestyle=':', alpha=0.6)

    # --- Calculate and Display Metrics ---

    # Use the mean estimates for calculating R2 and RMSE (standard practice)
    r2 = r2_score(gt_i, mean_i)
    rmse = np.sqrt(mean_squared_error(gt_i, mean_i))

    # Format the text box string
    metric_text = (
        f'$R^2$ Score: {r2:.4f}\n'
        f'RMSE: {rmse:.4f}'
    )

    # Add the text box to the right corner
    ax.text(
        0.98, 0.02,
        metric_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5')
    )

def plot_single_parameter_comparison_lite(ax, gt_i, mean_i, ci_i, ci_color, gt_color, main_color, confidence_level, param_name_i):
    # Determine plot limits based on min/max of both GT and mean estimates
    min_val = min(gt_i.min(), mean_i.min())
    max_val = max(gt_i.max(), mean_i.max())
    plot_range = max_val - min_val
    padding = plot_range * 0.1
    plot_min = min_val - padding
    plot_max = max_val + padding

    # --- Scatter of All Estimates (Optional, can be messy, often omitted for clarity) ---
    # Instead, we focus on mean and CI. If we plot all, the shape is (N_samples * N_estimates)
    # We can flatten the estimates for a general scatter plot:
    # flat_estimates = estimated_params[:, :, i].flatten()
    # flat_gt = np.repeat(gt_i, estimated_params.shape[1])
    # ax.scatter(flat_gt, flat_estimates, color='grey', alpha=0.1, label='All Estimates')

    # --- Confidence Intervals (Error Bars) ---
    ax.errorbar(
        gt_i,
        mean_i,
        yerr=ci_i,
        fmt='o',
        color=ci_color,
        alpha=0.6,
        capsize=3,
        label=fr'Mean $\pm$ {confidence_level * 100:.0f}% CI'
    )

    # --- Mean Comparison (Scatter) ---
    # Plot the mean values (on top of the error bars for better visibility)
    ax.scatter(gt_i, mean_i, color=main_color, marker='o', s=30, zorder=3, label='Mean Estimates')

    # --- Ground Truth Line (Y=X line) ---
    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        '--',
        color=gt_color,
        alpha=0.7,
        label='Ground Truth (Y=X)'
    )

    # --- Set Axes and Labels ---
    ax.set_title(f'{param_name_i}', fontsize=32)
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')  # Enforce square aspect ratio
    ax.grid(True, linestyle=':', alpha=0.6)


def plot_correlated_parameters(ax, gt_x, gt_y, est_x, est_y, name_x, name_y):
    ax.scatter(gt_x, gt_y, color="black", alpha=0.6, label="Ground Truth")
    ax.scatter(est_x, est_y, color="skyblue", alpha=0.7, label="Mean Estimates")

    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_title(f"Correlation: {name_x} vs {name_y}")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()

def plot_micro_correlated_parameters(ax, idx, gt_x, gt_y, est_x, est_y, name_x, name_y):
    ratio = gt_y / gt_x
    x_range = np.linspace(min(est_x), max(est_x), 100)
    line = ratio * x_range
    ax.scatter(est_x, est_y, alpha=0.1, s=10, color='purple', label='Posterior')
    ax.plot(x_range, line, 'k--', linewidth=2, label=f'{name_y}/{name_x} = {ratio:.2f}')
    ax.scatter([gt_x], [gt_y], c='gold', s=200, marker='*', edgecolors='k', label='Ground Truth', zorder=10)
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_title(f"Correlation: {name_x} vs {name_y}, id={idx}")
    ax.legend()
    sns.kdeplot(x=est_x, y=est_y, 
                cmap="Blues", 
                fill=True, 
                thresh=0.05,  # <--- Tweak this! Higher = more separation.
                levels=8,     # More levels = more detail on peaks
                alpha=0.8,    # Transparency of the fill
                cut=0,        # Don't smooth past the data limits
                label="Posterior", ax=ax)
    hb = ax.hexbin(est_x, est_y, gridsize=40, 
                   cmap='coolwarm', mincnt=2, bins='log') 
    return hb

def plot_micro_comparison(ax, idx, k_true, k_samples):
    """
    k_true: [k1, k3]
    k_samples: (N, 2)
    """
    # ax.scatter(k_samples[:, 0], k_samples[:, 1], 
    #     color='dodgerblue', s=5, alpha=0.2, label='Posteiror')
    
    sns.kdeplot(x=k_samples[:, 0], y=k_samples[:, 1], 
                cmap="Blues", 
                fill=True, 
                thresh=0.05,  # <--- Tweak this! Higher = more separation.
                levels=8,     # More levels = more detail on peaks
                alpha=0.8,    # Transparency of the fill
                cut=0,        # Don't smooth past the data limits
                label="Posterior", ax=ax)
    hb = ax.hexbin(k_samples[:, 0], k_samples[:, 1], gridsize=40, 
        cmap='coolwarm', mincnt=2, bins='log') 
    k1, k3 = k_true
    ax.plot(k1, k3, 'X', markersize=15, markeredgecolor='k', markerfacecolor='orange', label='Truth A (Target)')
    ax.plot(k3, k1, 'X', markersize=15, markeredgecolor='k', markerfacecolor='green', label='Truth B (Symmetric)')
    ax.set_xlabel(r"$k_1$")
    ax.set_ylabel(r"$k_3$")
    ax.set_title(f"Micro View: {idx}")
    return hb


def plot_parameter_regression_comparison(
        ground_truth_params: np.ndarray,
        estimated_params: np.ndarray,
        param_names: list[str],
        confidence_level: float = 0.95,
        n_rows=1, n_cols=3,
        filename=None,
        correlated=None,
        indices=None,
        wandb_log=False,
        lite=False
):
    if wandb_log:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import wandb
    else:
        import matplotlib.pyplot as plt

    N_samples, N_params = ground_truth_params.shape

    if estimated_params.shape[1] != N_samples or estimated_params.shape[2] != N_params:
        raise ValueError(
            f"Shape mismatch: Ground truth {ground_truth_params.shape} "
            f"vs. Estimates {estimated_params.shape}."
        )

    # Calculate Mean and Standard Deviation across estimates (N_samples axis)
    param_means = np.mean(estimated_params, axis=0)  # Shape (N_samples, N_params)
    param_stds = np.std(estimated_params, axis=0)  # Shape (N_samples, N_params)

    # Calculate the confidence interval margin (e.g., for 95% confidence, z-score is ~1.96)
    # Using scipy.stats.norm.ppf(1 - (1-confidence_level)/2) is more accurate,
    # but for simplicity, we'll use a hardcoded value (1.96 for 95% is a good proxy).
    z_score = 1.96 if confidence_level == 0.95 else 1.0  # Use 1 standard deviation as a fallback

    # CI margin = Z * (Std / sqrt(N_estimates))
    # We use Z * Std for a more generous confidence range often seen in these plots.
    ci_margin = z_score * param_stds

    # Calculate the number of rows/columns dynamically
    figsize = (n_cols * 4, n_rows * 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    # Flatten axes array for easy iteration, even if it's a 1D array
    if N_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    main_color = 'skyblue'
    ci_color = 'lightgray'
    gt_color = 'black'

    for i in range(N_params):
        ax = axes[i]
        gt_i = ground_truth_params[:, i]
        mean_i = param_means[:, i]
        ci_i = ci_margin[:, i]

        if lite:
            plot_single_parameter_comparison_lite(ax, gt_i, mean_i, ci_i, ci_color, gt_color, main_color, confidence_level, param_names[i])
        else:
            plot_single_parameter_comparison(ax, gt_i, mean_i, ci_i, ci_color, gt_color, main_color, confidence_level, param_names[i])

    
    if correlated is not None:
        idx, idy = correlated
        for i, (idx, idy) in enumerate(correlated):
            gt_x = ground_truth_params[:, idx]
            gt_y = ground_truth_params[:, idy]
            est_x = param_means[:, idx]
            est_y = param_means[:, idy]
            plot_correlated_parameters(axes[N_params+i], gt_x, gt_y, est_x, est_y, param_names[idx], param_names[idy])

    if indices is not None:
        if correlated is None:
            for i, idx in enumerate(indices):
                hb = plot_micro_comparison(axes[N_params+i], idx, ground_truth_params[idx], estimated_params[:, idx])
                data = hb.get_array()
                finite = data[np.isfinite(data)]
                if finite.size > 0 and finite.min() < finite.max():
                    hb.set_norm(mcolors.Normalize(vmin=finite.min(), vmax=finite.max()))
                    plt.colorbar(hb, ax=axes[N_params+i])
        else:
            for i, idx in enumerate(indices):
                for j, (_idx, _idy) in enumerate(correlated):
                    hb = plot_micro_correlated_parameters(axes[N_params+len(correlated)*(j+1)+i], idx, 
                         ground_truth_params[idx, _idx], ground_truth_params[idx, _idy],
                         estimated_params[:, idx, _idx], estimated_params[:, idx, _idy],
                         param_names[_idx], param_names[_idy])
                    data = hb.get_array()
                    finite = data[np.isfinite(data)]
                    if finite.size > 0 and finite.min() < finite.max():
                        hb.set_norm(mcolors.Normalize(vmin=finite.min(), vmax=finite.max()))
                        plt.colorbar(hb, ax=axes[N_params+len(correlated)*(j+1)+i])
        axes[-1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                        fancybox=True, shadow=False, ncol=3)

    # Hide unused subplots
    unused = [ax for ax in axes.ravel() if not has_content(ax)]
    if len(unused) > 0:
        for i in range(len(unused)):
            fig.delaxes(unused[i])

    plt.tight_layout()
    if wandb_log:
        wandb.log({
            "parameter_dist": wandb.Image(fig)
        })
    if filename is not None:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)