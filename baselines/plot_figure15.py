from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from scipy.stats import gaussian_kde

ORDER = ['SPEFM', 'IO', 'LORA', 'FROZEN']
COLORS = dict(SPEFM='#75a745', IO='#21a8d8', LORA='#e68b9b', FROZEN='#ef8b16')
LABELS = ['SPE-FM strategy', 'Input- and output-module\nfine-tuning',
          'LoRA fine-tuning', 'Frozen pretrained\nbackbone']


def plot(output, settings):
    output = Path(output)
    data = pd.read_csv(output / 'source_data.csv')
    font = font_manager.FontProperties(fname=font_manager.findfont(settings['font'])).get_name()
    matplotlib.rcParams.update({'font.family': font, 'font.size': 7.3, 'axes.labelsize': 8.1,
        'xtick.labelsize': 7.0, 'ytick.labelsize': 6.9, 'axes.linewidth': .72, 'axes.edgecolor': '#222222',
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(settings['width_mm'] / 25.4, settings['height_mm'] / 25.4),
        gridspec_kw=dict(left=.225, right=.985, bottom=.13, top=.97, wspace=.065, hspace=.42))
    ys = np.arange(4, 0, -1, dtype=float)
    text_artists = []
    for panel, (ax, metric, label) in enumerate(zip(axes.ravel(), ['RMSE', 'MAE', 'PICP', 'CWC'],
                                                  ['RMSE (mm)', 'MAE (mm)', 'PICP', 'CWC'])):
        arrays = {method: data.loc[data['method'] == method, metric].to_numpy(dtype=float) for method in ORDER}
        if any(not len(a) or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError('Missing or nonfinite node metrics.')
        maximum = max(float(a.max()) for a in arrays.values())
        xmin, xmax = (-.025, 1.045) if metric == 'PICP' else (-.025 * max(maximum, .1), 1.15 * max(maximum, .1))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(.48, 4.58)
        ax.set_yticks(ys)
        ax.set_yticklabels(LABELS if panel % 2 == 0 else [''] * 4)
        ax.tick_params(axis='both', direction='in', length=2.8, width=.65)
        ax.tick_params(axis='y', pad=4)
        ax.grid(True, color='#d1d1d1', linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel(label, labelpad=4)
        if metric == 'PICP':
            ax.set_xticks(np.linspace(0, 1, 6))
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, min_n_ticks=4))
            ax.set_xticks([v for v in ax.get_xticks() if xmin <= v <= xmax])
            ax.set_xlim(xmin, xmax)
        means = []
        for i, method in enumerate(ORDER):
            values, y, color = arrays[method], ys[i], COLORS[method]
            mean = float(values.mean())
            means.append(mean)
            grid = np.linspace(0, 1 if metric == 'PICP' else xmax, 700)
            if len(values) > 1 and float(values.std()) > 1e-10:
                kde = gaussian_kde(values, bw_method='scott')
                density = kde(grid) + kde(-grid)
                if metric == 'PICP':
                    density += kde(2 - grid)
                density = .44 * density / float(density.max())
                ax.fill_between(grid, y, y + density, color=color, alpha=.93, linewidth=0, zorder=2)
            else:
                ax.vlines(mean, y, y + .42, color=color, lw=1.4, zorder=2)
            rng = np.random.default_rng(20261016 + i)
            ax.scatter(values, y - .065 - rng.uniform(0, .16, len(values)), s=3.8,
                       facecolors=color, edgecolors='white', linewidths=.16, alpha=.78, zorder=4)
            right, dx = mean > xmin + .67 * (xmax - xmin), .022 * (xmax - xmin)
            text = ax.text(mean - dx if right else mean + dx, y + .105, f'Average {mean:.4f}',
                           ha='right' if right else 'left', va='center', fontsize=6.7, color='black', zorder=7)
            text_artists.append((ax, text))
        ax.plot(means, ys + .105, color='#ba2a39', linestyle=(0, (5, 2, 1.4, 2)), lw=.82, zorder=5)
        ax.scatter(means, ys + .105, s=12, c='#f1d2ae', edgecolors='none', zorder=6)
        ax.text(.5, -.25, f'({chr(97 + panel)})', transform=ax.transAxes, ha='center', va='center', fontsize=10.5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, text in text_artists:
        box, area = text.get_window_extent(renderer), ax.get_window_extent(renderer)
        if box.x0 < area.x0 + 1:
            text.set_ha('left')
            text.set_x(ax.get_xlim()[0] + .014 * np.ptp(ax.get_xlim()))
        elif box.x1 > area.x1 - 1:
            text.set_ha('right')
            text.set_x(ax.get_xlim()[1] - .014 * np.ptp(ax.get_xlim()))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes.ravel():
        for artist in [*ax.get_yticklabels(), *ax.get_xticklabels(), ax.xaxis.label, *ax.texts]:
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.x1 > fig.bbox.width + 1 or box.y0 < -1 or box.y1 > fig.bbox.height + 1:
                raise ValueError('Figure label exceeds the canvas.')
    stem = output / settings['stem']
    for suffix in ['.pdf', '.svg']:
        fig.savefig(str(stem) + suffix)
    fig.savefig(str(stem) + '.png', dpi=settings['dpi'])
    fig.savefig(str(stem) + '.tiff', dpi=settings['dpi'], pil_kwargs={'compression': 'tiff_lzw'})
    fig.savefig(str(stem) + '_preview.png', dpi=settings['preview_dpi'])
    plt.close(fig)
    (output / (settings['stem'] + '_caption.txt')).write_text(settings['caption'] + '\n', encoding='utf-8')
