import numpy as np

# 1-d cdf9-7 wavelet transform
def cdf97(row_vector):
    l = row_vector.shape[0]
    low_filter = [0.026748, -0.016864, -0.078223, 0.26686, 0.6029, 0.26686, -0.078223, -0.016864, 0.026748]
    high_filter = [0.04564, -0.02877, -0.29564, 0.55754, -0.29564, -0.02877, 0.04564]
    # add [0, 0, 0] to the begining and the end of the vector
    tmp_vector = np.insert(row_vector, 0, [0, 0, 0])
    row_vector = np.insert(tmp_vector, tmp_vector.shape[0], [0, 0, 0])
    low_vector = np.convolve(row_vector, low_filter, mode="same")
    high_vector = np.convolve(row_vector, high_filter, mode="same")
    return (low_vector, high_vector)

# 1-d inverse cdf9-7 wavelet transform
def inverse_cdf97(low_vector, high_vector):
    low_filter = [0.026748, -0.016864, -0.078223, 0.26686, 0.6029, 0.26686, -0.078223, -0.016864, 0.026748]
    high_filter = [0.04564, -0.02877, -0.29564, 0.55754, -0.29564, -0.02877, 0.04564]    
    lpr = np.multiply(high_filter, [-1, 1, -1, 1, -1, 1, -1]) * 2
    hpr = np.multiply(low_filter, [1, -1, 1, -1, 1, -1, 1, -1, 1]) * 2
    l = low_vector.shape[0] + high_vector.shape[0]
    tmp_low_vector = np.zeros((l))
    tmp_low_vector[::2] = low_vector
    tmp_high_vector = np.zeros((l))
    tmp_high_vector[1::2] = high_vector
    return np.convolve(tmp_low_vector, lpr, mode="same") + np.convolve(tmp_high_vector, hpr, mode="same")

# apply 1-d cdf9-7 wavelet transform to each row of a matrix
def cdf97_matrix(img_matrix):
    low_pass = None
    high_pass = None
    (h, _) = img_matrix.shape
    for i in range(h):
        (low_vector, high_vector) = cdf97(img_matrix[i, :])
        # for testing
        # tmp_row_vector = inverse_cdf97(low_vector[::2], high_vector[1::2])
        # print((tmp_row_vector[3:-3] - img_matrix[i, :]).max(), (tmp_row_vector[3:-3] - img_matrix[i, :]).min())
        if i == 0:
            low_pass = low_vector
            high_pass = high_vector
        else:
            low_pass = np.vstack([low_pass, low_vector])
            high_pass = np.vstack([high_pass, high_vector])
    return np.hstack([low_pass[:, ::2], high_pass[:, 1::2]])

# apply 1-d inverse cdf9-7 wavelet transform to each row of a matrix
def inverse_cdf97_matrix(wavelet_img):
    img_matrix = None
    (h, w) = wavelet_img.shape
    for i in range(h):
        row = inverse_cdf97(wavelet_img[i, 0:int(w/2)], wavelet_img[i, int(w/2):])
        if i ==0:
            img_matrix = row
        else:
            img_matrix = np.vstack([img_matrix, row])
    return img_matrix

# 2-d cdf9-7 wavelet transform
def cdf97_2d(img_matrix):
    # apply the wavelet transform to rows of a matrix
    img_1d = cdf97_matrix(img_matrix)
    # then to columns
    img_1d = img_1d.T
    img_2d = cdf97_matrix(img_1d)
    return img_2d.T

# 2-d inverse cdf9-7 wavelet transform
def inverse_cdf97_2d(wavelet_img):
    # apply the wavelet transform to columns
    img_1d = inverse_cdf97_matrix(wavelet_img.T)
    # then to rows
    img_1d = img_1d.T
    img_2d = inverse_cdf97_matrix(img_1d)
    return img_2d