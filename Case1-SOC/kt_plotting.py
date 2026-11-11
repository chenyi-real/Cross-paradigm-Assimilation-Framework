"""Times New Roman SOC plots, preserving original timestamps and predictions."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

CURVES = [('reference', 'Reference', '#333333', '-'),
          ('frozen_SOC', 'Data-driven SOC', '#45A8C0', (0, (6, 3))),
          ('corrected_SOC', 'KalmanNet-corrected SOC', '#F77676', (0, (3.5, 2)))]


def configure_font():
    for path in Path('/usr/share/fonts/truetype/msttcorefonts').glob('Times_New_Roman*.ttf'):
        font_manager.fontManager.addfont(str(path))
    try:
        resolved = font_manager.findfont('Times New Roman', fallback_to_default=False)
        family = 'Times New Roman'
    except ValueError:
        resolved = font_manager.findfont('DejaVu Serif')
        family = 'DejaVu Serif'
    plt.rcParams.update({'font.family':family, 'font.size':13,
                         'axes.titlesize':15, 'axes.labelsize':14,
                         'xtick.labelsize':12, 'ytick.labelsize':12,
                         'mathtext.fontset':'stix', 'pdf.fonttype':42,
                         'ps.fonttype':42, 'axes.unicode_minus':False})
    return resolved


def draw_soc(ax, trace, title=None):
    time = trace.time_s.to_numpy()
    for col,label,color,style in CURVES:
        ax.plot(time,trace[col],label=label,color=color,ls=style,
                lw=2. if col=='reference' else 1.6)
    values = trace[['reference','frozen_SOC','corrected_SOC']].to_numpy()
    span = max(float(np.ptp(values)),.01)
    ax.set_ylim(max(-.02,float(values.min())-.09*span),
                min(1.02,float(values.max())+.12*span))
    ax.set_ylabel('SOC')
    ax.ticklabel_format(axis='both',style='plain',useOffset=False)
    ax.grid(alpha=.18)
    ax.set_xlim(float(time[0]),float(time[-1]))
    if title:
        ax.set_title(title)


def save(fig,directory,stem):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:
        fig.savefig(directory/f'{stem}.{ext}',dpi=300)
    plt.close(fig)


def plot_trace(trace,family,unit,directory,max_seconds=0,*,with_error=False):
    configure_font()
    if max_seconds>0:
        trace = trace[trace.time_s<=trace.time_s.iloc[0]+max_seconds].copy()
    if len(trace)<2:
        raise ValueError('At least two real observations are required for a plot')
    rows = 3 if with_error else 2
    ratios = [2.4,1.,1.] if with_error else [3.,1.]
    fig,axes = plt.subplots(rows,1,figsize=(8.6,7.2 if with_error else 5.8),
                            sharex=True,layout='constrained',
                            gridspec_kw={'height_ratios':ratios})
    draw_soc(axes[0],trace,f'{family}: {unit}')
    axes[0].legend(frameon=False,ncols=3,fontsize=10,loc='best')
    if with_error:
        for col,label,color,style in CURVES[1:]:
            axes[1].plot(trace.time_s,np.abs(trace[col]-trace.reference),
                         color=color,ls=style,lw=1.5,label=label)
        axes[1].set_ylabel('Absolute SOC error')
        axes[1].ticklabel_format(axis='y',style='plain',useOffset=False)
        axes[1].grid(alpha=.18)
    used = trace[trace.scored] if 'scored' in trace else trace.iloc[1:]
    axes[-1].plot(used.time_s,used.K,color='#3C6EB9',lw=1.3)
    axes[-1].set_ylabel('K(t)')
    axes[-1].set_xlabel('Time (s)')
    upper = min(1.,max(.05,float(used.K.max())*1.1)) if len(used) else 1.
    axes[-1].set_ylim(-.025*upper,1.025*upper)
    axes[-1].grid(alpha=.18)
    axes[-1].ticklabel_format(axis='both',style='plain',useOffset=False)
    save(fig,directory,'SOC-and-Kt')


def plot_soc_only(trace,family,unit,directory):
    configure_font()
    fig,ax = plt.subplots(figsize=(8.6,4.5),layout='constrained')
    draw_soc(ax,trace,f'{family}: {unit}')
    ax.set_xlabel('Time (s)')
    ax.legend(frameon=False,ncols=3,fontsize=10,loc='best')
    save(fig,directory,'SOC-three-curves')


def plot_summary(selected,directory):
    configure_font()
    fig,axes = plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ax,(family,unit,trace) in zip(axes.flat,selected):
        draw_soc(ax,trace,f'{family}: {unit}')
        ax.set_xlabel('Time (s)')
        ax.legend(frameon=False,fontsize=9,loc='best')
    save(fig,directory,'Four-datasets-selected-SOC')
