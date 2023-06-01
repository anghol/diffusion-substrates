from enum import Enum

class Color(Enum):
    WHITE = 255
    BLACK = 0

class Filter(Enum):
    AVERAGE = 'Average'
    MEDIAN = 'Median'
    GAUSSIAN = 'Gaussian'

class Morph(Enum):
    EROSION = 'Erosion'
    DILATION = 'Dilation'
    OPENING = 'Opening'
    CLOSING = 'Closing'

class Edges_Alg(Enum):
    SOBEL = 'Sobel'
    CANNY = 'Canny'

class Mask_Alg(Enum):
    BLOB_DETECT = 'Blob detect'