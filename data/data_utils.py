import numpy as np
import torch


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


def generate_gaussian_heatmap(cfg, h, w, bbox_center_int, threshold, set_constant_value=0, normalise=False):
    # This will generate a gaussian map in the output dimension size
    object_heatmap = np.zeros((cfg["heatmap"]["output_dimension"],
                               cfg["heatmap"]["output_dimension"]))

    output_height = output_width = cfg["heatmap"]["output_dimension"]

    gaussian_radius, gaussian_peak = generate_gaussian_peak(cfg, h, w)
    if (set_constant_value != 0):
        if (normalise):
            constant_value = set_constant_value / cfg["heatmap"]["output_dimension"]
        else:
            constant_value = set_constant_value
        gaussian_peak[gaussian_peak < threshold] = 0
        gaussian_peak[gaussian_peak >= threshold] = constant_value

    left, right = min(bbox_center_int[0], gaussian_radius), min(output_width - bbox_center_int[0],
                                                                gaussian_radius + 1)
    top, bottom = min(bbox_center_int[1], gaussian_radius), min(output_height - bbox_center_int[1],
                                                                gaussian_radius + 1)

    masked_object_heatmap = object_heatmap[bbox_center_int[1] - top:bbox_center_int[1] + bottom,
                            bbox_center_int[0] - left:bbox_center_int[0] + right]
    masked_gaussian_peak = gaussian_peak[gaussian_radius - top:gaussian_radius + bottom,
                           gaussian_radius - left:gaussian_radius + right]

    np.maximum(masked_object_heatmap, masked_gaussian_peak, out=masked_object_heatmap)
    return object_heatmap


def create_heatmap_object(cfg, heatmap_bounding_box):
    # [x1,y1,w,h] -> [x1,y1,x1+w,y1+h]
    bbox = np.array([heatmap_bounding_box[0], heatmap_bounding_box[1],
                     heatmap_bounding_box[0] + heatmap_bounding_box[2],
                     heatmap_bounding_box[1] + heatmap_bounding_box[3]],
                    dtype=np.float32)
    # [x_center, y_center]
    bbox_center = np.array(
        [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.int32)
    # [h,w]
    bbox_h, bbox_w = heatmap_bounding_box[3], heatmap_bounding_box[2]
    center_heatmap = generate_gaussian_heatmap(cfg, bbox_h, bbox_w, bbox_center, threshold=0.5, set_constant_value=1)
    # , normalise=False)
    bbox_heatmap_w = generate_gaussian_heatmap(cfg, bbox_h, bbox_w, bbox_center, threshold=0.5,
                                               set_constant_value=bbox_w,
                                               normalise=False)
    bbox_heatmap_h = generate_gaussian_heatmap(cfg, bbox_h, bbox_w, bbox_center, threshold=0.5,
                                               set_constant_value=bbox_h,
                                               normalise=False)
    bbox_heatmap = np.vstack((np.expand_dims(bbox_heatmap_w, axis=0), np.expand_dims(bbox_heatmap_h, axis=0)))
    return center_heatmap, bbox_heatmap, bbox_center
