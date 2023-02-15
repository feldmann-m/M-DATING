import library.io, library.plot
import numpy as np

myfinaldata, flags = library.io.read_del_data("meso_example/DVL/DVL2217907457L.806")

from matplotlib.pyplot import cm

resolution=0.5
r, theta =np.meshgrid(np.arange(0, myfinaldata.shape[1] *resolution, resolution),
                            np.radians(np.linspace(0, 360, 360)))

print(myfinaldata.shape)

library.plot.plot_ppi_MF_masked(theta=theta,
    myfinaldata=myfinaldata, r=r,
    vmin=-33, vmax=+33, cmap=cm.seismic, bound=np.arange(-33, +33),
    imtitle="test", imname="test.png", savepath="out"
    )
