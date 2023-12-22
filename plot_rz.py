"""
Plot planes from joint analysis files.

Usage:
    plot_snapshots.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dedalus.extras import plot_tools


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ['u(phi=0)', 'u1(phi=0)', 'u(phi=pi)', 'u1(phi=pi)']
    scale = 2
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Layout
    nrows, ncols = 2, 1
    image = plot_tools.Box(7.5, 1)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.2, 0.1, 0.1, 0.1)
    clims = {'u(phi=0)': (0,1), 'u1(phi=0)':(-0.5,0.5), 'u(phi=pi)': (0,1), 'u1(phi=pi)':(-0.5,0.5)}
    flip = lambda xmesh, ymesh, data: (xmesh, -ymesh, data)

    def get_axis(i,j):
        return mfig.add_axes(i, j, [0, 0, 1, 1])

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    mfig_axes = [mfig.add_axes(0, 0, [0, 0, 1, 1]), mfig.add_axes(1, 0, [0, 0, 1, 1])]
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                if n > 1:
                    func = flip
                else:
                    func = None
                # Build subfigure axes
                axes = mfig_axes[n % 2]
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]
                slices = (index,0,slice(None),0,slice(None))
                plot_tools.plot_bot(dset, (2,4), slices, axes=axes, title=task, clim=clims[task], func=func)
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.45, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

