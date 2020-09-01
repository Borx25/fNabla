import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import pandas as pd

shape = (256, 256)
kernel = np.full(shape, 0.0 + 0.0j, dtype=np.complex64)
kernel[0][1] = -2. + 0.j
kernel[0][-1] = 2. + 0.j
kernel[1][0] = 0. + 2.j
kernel[1][1] = -1. + 1.j
kernel[1][-1] = 1. + 1.j
kernel[-1][0] = 0. - 2.j
kernel[-1][1] = -1. - 1.j
kernel[-1][-1] = 1. - 1.j
operator = np.fft.fft2(kernel)

freq = np.full(shape, 0.0 + 0.0j, dtype=np.complex64)

for i in range(0, shape[0]):  # rows
    for j in range(0, shape[1]):  # cols
        freq[i, j] = 2.0 * np.pi * ((j / shape[1] - (2 * j) // shape[1]) * 1j + (i / shape[0] - (2 * i) // shape[0]))

x = np.linspace(0.0, 1.0, shape[0]).reshape(shape[0], 1)
y = np.linspace(0.0, 1.0, shape[1]).reshape(1, shape[1])

# polynomial features
features = {}
degrees = 8
for i in range(degrees):
    for j in range(degrees):
        features['x^{}*y^{}'.format(i, j)] = np.matmul(x ** i, y ** j).flatten()

dataset = pd.DataFrame(features)

reg_real = LinearRegression().fit(dataset.values, operator.real.flatten())
coefficients_real = reg_real.coef_.reshape(-1, 1)
# reconstruct = np.matmul(dataset.values, coefficients_real).reshape(shape[0], shape[1])
reg_imag = LinearRegression().fit(dataset.values, operator.imag.flatten())
coefficients_imag = reg_imag.coef_.reshape(-1, 1)

op_real = np.full(shape, 0.0, dtype=np.float64)
for i in range(shape[0]):
    for j in range(shape[1]):
        x_coor = i / shape[0]
        y_coor = j / shape[1]
        result = 0
        for c_x in range(degrees):
            for c_y in range(degrees):
                result += coefficients_real[c_x * degrees + c_y] * x_coor ** c_x * y_coor ** c_y
        op_real[i, j] = result

op_imag = np.full(shape, 0.0, dtype=np.float64)
for i in range(shape[0]):
    for j in range(shape[1]):
        x_coor = i / shape[0]
        y_coor = j / shape[1]
        result = 0
        for c_x in range(degrees):
            for c_y in range(degrees):
                result += coefficients_imag[c_x * degrees + c_y] * x_coor ** c_x * y_coor ** c_y
        op_imag[i, j] = result

px.imshow(np.fft.fftshift(op_real)).show()
px.imshow(np.fft.fftshift(op_imag)).show()
