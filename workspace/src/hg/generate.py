from random import sample
import numpy as np
import matplotlib.pyplot as plt

def generate_hg(sample_num: int, g: float):
    g2 = g * g
    samples = np.random.rand(sample_num) * 2.0 - 1.0
    cos_t =  (1. + g2 - ((1. - g2) / (1. - g + 2. * g * (-abs(samples) + 1.0)))**2) / 2. / g
    sin_t = np.sqrt(np.maximum(1. - cos_t ** 2, np.zeros_like(cos_t))) * np.sign(samples)
    return cos_t, sin_t

def bunny_gen_hg(sample_num: int, g: float):
    samples = np.random.rand(sample_num) * 2.0 - 1.0
    cos_t = np.cos(2 * np.arctan((1 - g) / (1 + g) * np.tan(np.pi / 2 * (samples * 1.999999 - 0.99999999))))
    sin_t = np.sqrt(np.maximum(1. - cos_t ** 2, np.zeros_like(cos_t))) * np.sign(samples)
    return cos_t, sin_t

def rayleigh_gen(sample_num):
    def rayleigh_strength(x: np.ndarray) -> np.ndarray:
        return 3 / (16 * np.pi) * (1 + x ** 2)
    angles = np.random.rand(sample_num) * 2 * np.pi - np.pi
    cos_ts = np.cos(angles)
    sin_ts = np.sin(angles)
    length = rayleigh_strength(cos_ts)
    return np.stack((cos_ts * length, sin_ts * length), axis = 1)
    
def hg_phase_test_main():
    csamples, ssamples = generate_hg(280000, 0.9)
    b_csamples, b_ssamples = bunny_gen_hg(280000, 0.9)

    plt.figure(0)
    plt.subplot(3, 1, 1)
    plt.hist(csamples, bins = np.linspace(-1, 1, 500))
    plt.subplot(3, 1, 2)
    plt.hist(ssamples, bins = np.linspace(-1, 1, 500))
    plt.subplot(3, 1, 3)
    plt.hist(csamples ** 2, bins = np.linspace(-1, 1, 500))

    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.hist(b_csamples, bins = np.linspace(-1, 1, 500))
    plt.subplot(3, 1, 2)
    plt.hist(b_ssamples, bins = np.linspace(-1, 1, 500))
    plt.subplot(3, 1, 3)
    plt.hist(b_csamples ** 2, bins = np.linspace(-1, 1, 500))

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.scatter(ssamples, csamples, s = 3.0, alpha = 0.005)
    plt.subplot(1, 2, 2)
    plt.scatter(b_ssamples, b_csamples, s = 3.0, alpha = 0.002)

    plt.show()
    
def rayleigh_plotting():
    samples = rayleigh_gen(65536)
    plt.scatter(samples[:, 0], samples[:, 1], alpha = 0.01, s = 1)
    plt.grid(axis = 'both')
    plt.show()

if __name__ == '__main__':
   rayleigh_plotting()