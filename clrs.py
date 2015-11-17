import numpy as np
import colorsys

def get_colors(num_colors):
    colors=[]
    rar = np.arange(0., 360., 360. / num_colors)
    np.random.shuffle(rar)
    for i in rar:
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors
