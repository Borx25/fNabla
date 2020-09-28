import numpy as np
from PIL.Image import fromarray, open as pillow_open
from scipy.ndimage import rotate as nd_rotate, gaussian_filter, fourier_gaussian

version = 0.1


class FMath:
    @staticmethod
    def sigmoid(x):
        return 2 / (1 + np.exp(-x)) - 1

    @staticmethod
    def integrate(gradient, slope_range):
        freq_x = np.fft.fftfreq(gradient.shape[0])
        freq_y = np.fft.fftfreq(gradient.shape[1])
        integrator = np.add.outer(freq_x, freq_y) / (1j * (np.add.outer(freq_x ** 2.0, freq_y ** 2.0) + np.finfo(float).eps))
        integrator -= fourier_gaussian(integrator, sigma=slope_range * gradient.shape[0])
        return np.fft.ifft2(np.fft.fft2(gradient) * integrator).real.astype(np.float32)

    @staticmethod
    def equalize_0_to_1(arr):
        arr -= np.amin(arr)
        arr /= np.amax(arr)
        return arr


class MeshMap:
    suffix = 'MeshMap'
    name = 'Generic MeshMap'
    id = -1
    channels = 'RGB'

    def __init__(self):
        self.data = None

    @classmethod
    def get_suffix(cls):
        return cls.suffix

    def export(self, output):
        fromarray(np.uint8(np.rint((self.data + 1.0) * 127.5))).save('{}_{}.png'.format(output, self.get_suffix()))

    @staticmethod
    def open(file, mesh_map_class, *args, **kwargs):
        if mesh_map_class.channels == 'F':
            data = FMath.equalize_0_to_1((np.array(pillow_open(file).convert(mesh_map_class.channels), dtype=np.float32)))
        else:
            data = (np.array(pillow_open(file).convert(mesh_map_class.channels), dtype=np.float32) / 127.5) - 1.0
        return mesh_map_class(data, *args, **kwargs)

    def compute(self, mesh_map_class, *args, **kwargs):
        funcs = [self.compute_height, self.compute_normal, self.compute_curvature, self.compute_ao]
        return funcs[mesh_map_class.id](*args, **kwargs)

    def compute_normal(self):
        return self

    def compute_height(self):
        return self

    def compute_ao(self):
        return self

    def compute_curvature(self):
        return self


class HeightMap(MeshMap):
    suffix = 'height'
    name = 'Height'
    id = 0
    channels = 'F'

    def __init__(self, data):
        super().__init__()
        self.data = data

    def export(self, output):
        fromarray(self.data, mode='F').save('{}_{}.tiff'.format(output, self.get_suffix()))

    def compute_normal(self, intensity=1.0, swizzle_coordinates=(1, 1, 1)):
        arr = np.copy(self.data)
        y, x = np.gradient(arr*100*intensity)
        z = np.ones_like(x)

        return NormalMapTS(np.dstack((-x, swizzle_coordinates[1] * y, z)), swizzle_coordinates)

    def compute_ao(self, normal_map=None, intensity=1.0, distance=0.25, samples=8, facing_attenuation=0.15):
        arr = np.copy(self.data)
        if normal_map is None:
            normal_map = self.compute_normal(self.data)

        sigmas = np.geomspace(start=1.0, stop=np.amax(arr.shape) * distance, num=int(samples), dtype=float)
        fft_height = np.fft.fft2(arr)
        sample_occlusion = sum([np.fft.ifft2(fourier_gaussian(fft_height, sigma=sigma)).real - arr for sigma in sigmas]) / samples
        sample_occlusion = FMath.equalize_0_to_1(sample_occlusion)

        ambient_occlusion = 1.0 - sample_occlusion
        facing_factor = normal_map.facing_normal() * -1.0 * facing_attenuation + (1.0 - facing_attenuation)
        ambient_occlusion_combined = np.clip(ambient_occlusion / facing_factor, 0.0, 1.0)
        ambient_occlusion_combined = 1.0 - ((1.0 - ambient_occlusion_combined) * (1.0 - ambient_occlusion_combined))
        ambient_occlusion_combined **= intensity

        ambient_occlusion_combined = ambient_occlusion_combined * 2.0 - 1.0
        return AmbientOcclusionMap(ambient_occlusion_combined)

    def compute_curvature(self, *args, **kwargs):
        return self.compute_normal().compute_curvature(*args, **kwargs)


class NormalMapTS(MeshMap):
    suffix = 'normal'
    name = 'Tangent Space Normal'
    id = 1
    channels = 'RGB'

    def __init__(self, data, swizzle_coordinates=None):
        super().__init__()
        self.data = data
        if swizzle_coordinates is None:
            swizzle_coordinates = (1, 1, 1)
        else:
            swizzle_coordinates = (x / abs(x) for x in swizzle_coordinates)
        self.__sw_x, self.__sw_y, self.__sw_z = swizzle_coordinates
        self.normalize()

    def set_swizzle_coordinates(self, new):
        self.data[:, :, 0] *= self.__sw_x * new[0]
        self.data[:, :, 1] *= self.__sw_y * new[1]
        self.data[:, :, 2] *= self.__sw_z * new[2]
        self.__sw_x, self.__sw_y, self.__sw_z = new

    def rot90(self, k=1):
        k = int(k) % 4
        rotated = np.rot90(self.data, k)
        arr = np.copy(rotated)
        if k == 1:  # 90º CW => R:-G, G:R
            arr[:, :, 0] = -rotated[:, :, 1]
            arr[:, :, 1] = rotated[:, :, 0]
        elif k == 2:  # 180º => R:-R, G:-G
            arr[:, :, 0] = -rotated[:, :, 0]
            arr[:, :, 1] = -rotated[:, :, 1]
        elif k == 3:  # 270º CW => R:G, G:-R
            arr[:, :, 0] = rotated[:, :, 1]
            arr[:, :, 1] = -rotated[:, :, 0]
        return NormalMapTS(arr, swizzle_coordinates=(self.__sw_x, self.__sw_y, self.__sw_z))

    def rotate(self, angle=0.0, turns=0.0, reshape=False):
        # IMPROVEMENT: si divide perfecto en 90 callear aqui rot90 con el k que haga falta en vez de todas las cuentas
        if angle != 0.0:
            turns = angle / 360
        turns = -turns % 1.0
        arr = nd_rotate(self.data, turns * 360, reshape=reshape, mode='wrap')
        swizzle = [self.__sw_x, -self.__sw_y, self.__sw_z]
        arr *= swizzle
        xy = np.copy(arr[:, :, :2])
        arr[:, :, 0] = xy @ [np.cos(2 * np.pi * turns), np.sin(2 * np.pi * turns)]
        arr[:, :, 1] = xy @ [-np.sin(2 * np.pi * turns), np.cos(2 * np.pi * turns)]
        arr *= swizzle
        return NormalMapTS(arr, swizzle_coordinates=(self.__sw_x, self.__sw_y, self.__sw_z))

    def flip_h(self):
        arr = np.flip(np.copy(self.data), 1)
        arr[:, :, 0] = -arr[:, :, 0]
        return NormalMapTS(arr, swizzle_coordinates=(self.__sw_x, self.__sw_y, self.__sw_z))

    def flip_v(self):
        arr = np.flip(np.copy(self.data), 0)
        arr[:, :, 1] = -arr[:, :, 1]
        return NormalMapTS(arr, swizzle_coordinates=(self.__sw_x, self.__sw_y, self.__sw_z))

    def normalize(self):
        self.data /= np.linalg.norm(self.data, axis=-1, keepdims=True)

    def facing_normal(self):
        arr = -(np.abs(np.copy(self.data[:, :, :2])) * 2.0 - 1.0)
        facing_normal = (arr[:, :, 0] / 2 + 0.5) * (arr[:, :, 1] / 2 + 0.5)
        return facing_normal

    def compute_height(self, slope_range=1.0):
        arr = np.copy(self.data)
        return HeightMap(FMath.equalize_0_to_1(FMath.integrate(-arr[:, :, 0], slope_range) + \
                                               np.rot90(FMath.integrate(self.__sw_y * np.rot90(arr[:, :, 1]), slope_range), 3)
                                               ))

    def compute_curvature(self, intensity=1.0, smoothing=0.5):
        arr = np.copy(self.data)
        if smoothing > 0.0:
            arr = gaussian_filter(arr, sigma=smoothing, mode='wrap')

        x = np.gradient(arr[:, :, 0], axis=1)
        y = np.gradient(-self.__sw_y * arr[:, :, 1], axis=0)

        return CurvatureMap(FMath.sigmoid((x + y) * intensity))

    def compute_ao(self, *args, **kwargs):
        return self.compute_height().compute_ao(normal_map=self, *args, **kwargs)


class CurvatureMap(MeshMap):
    suffix = 'curvature'
    name = 'Curvature'
    id = 2
    channels = 'L'

    def __init__(self, data):
        super().__init__()
        self.data = data

    def compute_normal(self, intensity=1.0, swizzle_coordinates=(1, 1, 1), slope_range=1.0, flatten=0.5):
        arr = np.copy(self.data) * 0.5 * intensity
        x = FMath.integrate(-1.0 * arr, slope_range)
        y = np.rot90(FMath.integrate(np.rot90(arr), slope_range), 3)
        z = np.ones_like(x)

        flatten_factor = np.maximum(np.abs(x), np.abs(y)) ** flatten
        x, y = x * flatten_factor, y * flatten_factor

        return NormalMapTS(np.dstack((-x, -swizzle_coordinates[1] * y, z)), swizzle_coordinates)

    def compute_height(self, slope_range=1.0):
        return self.compute_normal(slope_range=slope_range).compute_height(slope_range=slope_range)

    def compute_ao(self, slope_range=1.0, *args, **kwargs):
        normal = self.compute_normal(slope_range=slope_range)
        return normal.compute_height(slope_range=slope_range).compute_ao(normal_map=normal, *args, **kwargs)


class AmbientOcclusionMap(MeshMap):
    suffix = 'ao'
    name = 'Ambient Occlusion'
    id = 3
    channels = 'L'

    def __init__(self, data):
        super().__init__()
        self.data = data
