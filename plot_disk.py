"""
Plot disk outputs.

Usage:
    plot_disk.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames_disk]

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
    tasks = ['u(z=0)', 'u1(z=0)']
    cmap = plt.cm.RdBu_r
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    title_func = lambda sim_time: 't = {:.1f}'.format(np.round(sim_time,1))
    dpi = 150
    func = lambda phi, r, data: (r*np.cos(phi), r*np.sin(phi), data)
    clims = {'u(z=0)': (0,1), 'u1(z=0)':(-0.5,0.5)}

    # Layout
    nrows, ncols = 1, 2
    image = plot_tools.Box(1, 1)
    pad = plot_tools.Frame(0.1, 0.1, 0.1, 0.1)
    margin = plot_tools.Frame(0.2, 0.1, 0.1, 0.1)
    scale = 3

    # Create figure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    # Plotting loop
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]
                slices = (index, 0, 0, slice(None), slice(None))
                paxes, caxes = plot_tools.plot_bot(dset, (3,4), slices, axes=axes, title=task, clim=clims[task], visible_axes=False, func=func, cmap=cmap)
                paxes.axis('off')
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.5, y=title_height, ha='center')
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

