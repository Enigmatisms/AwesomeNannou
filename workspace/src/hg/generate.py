import numpy as np
import matplotlib.pyplot as plt

def generate(sample_num: int, g: float):
    g2 = g * g
    samples = np.random.rand(sample_num) * 2.0 - 1.0
    cos_t =  (1. + g2 - ((1. - g2) / (1. - g + 2. * g * (-abs(samples) + 1.0)))**2) / 2. / g
    sin_t = np.sqrt(np.maximum(1. - cos_t ** 2, np.zeros_like(cos_t))) * np.sign(samples)
    return cos_t, sin_t

def bunny_gen(sample_num: int, g: float):
    samples = np.random.rand(sample_num) * 2.0 - 1.0
    cos_t = np.cos(2 * np.arctan((1 - g) / (1 + g) * np.tan(np.pi / 2 * (samples * 1.999999 - 0.99999999))))
    sin_t = np.sqrt(np.maximum(1. - cos_t ** 2, np.zeros_like(cos_t))) * np.sign(samples)
    return cos_t, sin_t
    

if __name__ == '__main__':
    csamples, ssamples = generate(280000, 0.9)
    b_csamples, b_ssamples = bunny_gen(280000, 0.9)

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