import numpy as np


def get_gaussian_radius_centernet(height, width, min_overlap=0.5):
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return int(min(r1, r2, r3))


def get_gaussian_radius(cfg, height, width):
    if cfg["heatmap"]["fix_radius"]:
        r = cfg["heatmap"]["fix_radius_value"]
    else:
        radius_scale = cfg["heatmap"]["radius_scaling"]
        r = np.sqrt(height ** 2 + width ** 2)
        r = r / radius_scale

    return int(r)


def generate_gaussian_peak(cfg, height, width):
    # This will only generate a matrix of size [diameter, diameter] that has gaussian distribution
    gaussian_radius = get_gaussian_radius(cfg, height, width)
    gaussian_diameter = 2 * gaussian_radius + 1
    sigma = gaussian_diameter / 6
    m, n = [(ss - 1.) / 2. for ss in (gaussian_diameter, gaussian_diameter)]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    gaussian_peak = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # gaussian_peak /= gaussian_peak.max()
    gaussian_peak[gaussian_peak < 1e-3] = 0
    return gaussian_radius, gaussian_peak
