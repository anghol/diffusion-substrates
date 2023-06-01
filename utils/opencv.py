import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from utils.consts import * 


def delete_description(image: cv.Mat, size: int) -> cv.Mat:
    height = image.shape[0]
    image_with_deleted_description = image[:height-size, :]
    return image_with_deleted_description


def rgb_to_gray(image: cv.Mat) -> cv.Mat:
    if len(image.shape) != 2: image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else: image_gray = image
    return image_gray


def get_size_for_crop(image: cv.Mat) -> int:
    height = image.shape[0]
    image_gray = rgb_to_gray(image)

    # intesity of pixels on the y axis
    intensity_y = image_gray[:, 0]

    # find y-coord where black description bar starts
    for i in range(height-2, -1, -1):
        if intensity_y[i] - intensity_y[i+1] > 0:
            start_description = i+1
            break
    
    return height - start_description


def filter_image(image: cv.Mat, filter: Filter, average_ksize=5, median_ksize=7,
                gauss_ksize=5, x_deviation=1, y_deviation=1) -> cv.Mat:

    if filter == Filter.AVERAGE:
        filtered_image = cv.blur(image, (average_ksize, average_ksize))

    elif filter == Filter.MEDIAN:
        if median_ksize % 2 != 1:
            median_ksize = 7
        filtered_image = cv.medianBlur(image, median_ksize)

    elif filter == Filter.GAUSSIAN:
        if gauss_ksize % 2 != 1:
            gauss_ksize = 5
        filtered_image = cv.GaussianBlur(image, (gauss_ksize, gauss_ksize), sigmaX=x_deviation, sigmaY=y_deviation)

    else:
        filtered_image = image

    return filtered_image


def plot_intensity_dist(image: cv.Mat, bins: int, figsize: tuple):
    image_gray = rgb_to_gray(image)

    plt.figure(figsize=figsize)
    plt.title('Intensity histogram')
    ax = plt.hist(image_gray.ravel(), bins, range=(0, 256))
    # ax = cv.calcHist([image_gray], [0], None, [bins], [0, 256])
    # plt.plot(ax)
    plt.show()

    return ax


def morph_transform(image: cv.Mat, operation: Morph, ksize=5, iters=1):

    kernel = np.ones((ksize, ksize), np.uint8)

    if operation == Morph.EROSION:
        transformed_image = cv.erode(image, kernel, iterations=iters)
    
    elif operation == Morph.DILATION:
        transformed_image = cv.dilate(image, kernel, iterations=iters)
    
    elif operation == Morph.OPENING:
        transformed_image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=iters)

    elif operation == Morph.CLOSING:
        transformed_image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=iters)

    else:
        transformed_image = image

    return transformed_image


def detect_edges(image: cv.Mat, algorithm: Edges_Alg, gauss_ksize=3, sobel_ksize=5,
                dx=1, dy=1, treshold1=100, treshold2=200):
    
    if algorithm == Edges_Alg.SOBEL:
        image_gray = rgb_to_gray(image)

        if sobel_ksize not in (1, 3, 5, 7):
            sobel_ksize = 5

        blur_params = {'gauss_ksize': gauss_ksize, 'x_deviation': 0, 'y_deviation': 0}
        image_blur = filter_image(image_gray, Filter.GAUSSIAN, **blur_params)
        edged_image = cv.Sobel(image_blur, cv.CV_64F, dx=dx, dy=dy, ksize=sobel_ksize)
    
    elif algorithm == Edges_Alg.CANNY:
        edged_image = cv.Canny(image, treshold1, treshold2)

    else:
        edged_image = image
    
    return edged_image