#!/usr/bin/env python
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

class Echelle(object):
    def __init__(self):
        self.data = None

    def loadimage(self, filename):
        hdu = fits.open(filename)
        self.data = hdu[0].data
        # normalization
        self.data = 1.0*self.data - np.min(self.data)
        self.data /= np.max(self.data)

    def saveimage(self, filename):
        hdu = fits.PrimaryHDU(self.data)
        hdu.writeto(filename, overwrite=True)

    def artificial_flat(self):
        numrows, numcols = self.data.shape
        x = np.arange(numcols)
        y = np.arange(numrows)
        X, Y = np.meshgrid(x, y, copy=False)

        Z = np.copy(self.data)
        # Z -= np.min(Z)
        # Z /= np.max(Z)
        # Z += 1

        i = np.where(self.data > np.mean(self.data))
        X = X[i]
        Y = Y[i]
        Z = Z[i]
        # Z = X ** 2 + Y ** 2 + np.random.rand(*X.shape) * 0.01

        X = X.flatten()
        Y = Y.flatten()

        A = np.array([X * 0 + 1, X, Y, X ** 2, Y ** 2, X * Y]).T
        B = Z.flatten()

        coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=1e-5)
        X, Y = np.meshgrid(x, y, copy=False)
        flat = coeff[0] + coeff[1]*X + coeff[2]*Y + coeff[3]*X**2 + coeff[4]*Y**2 + coeff[5]*X*Y
        # normalize the flat since we get negative values
        # flat -= np.min(flat)
        # flat /= np.max(flat)
        # plt.imshow(flat)
        # plt.show()
        # plt.close()
        hdu = fits.PrimaryHDU(flat)
        hdu.writeto('/Users/jwe/Projects/Astronomie/Spectra/artificial_flat.fit', overwrite=True)
        # self.data -= np.min(self.data)
        # self.data /= np.max(self.data)

        self.data /= flat
        #
        # self.data /= np.nanmax(self.data)
        plt.imshow(self.data, vmin=-1.0, vmax=1.0)
        plt.show()
        plt.close()

    def extract_orders(self):
        numrows, numcols = self.data.shape
        # cols = 300
        c = np.zeros(numcols)
        cols = np.arange(numcols)
        window = 25
        reference = self.data[:, 0]
        for i in range(1, numcols):
            cv = np.correlate(self.data[:, i], reference, mode='same')
            mid = cv.shape[0] // 2
            prediction = int(c[i - 1])
            # tau is empirical and must be below 0.05
            tau = 0.01
            # shift with empty filling would be better, but roll should do for small window sizes
            reference = (1.0 - tau) * reference + tau * np.roll(self.data[:, i-1], int(-c[i-1]))
            c[i] = np.argmax(cv[mid - window + prediction: mid + window + prediction]) - window + prediction
            # print('{:3d} {:3.0f} {:3d}'.format(i, c[i], prediction))
            # plt.plot(np.arange(-window, window), cv[mid - window + p: mid + window + p])
        # plt.show()
        # plt.grid()
        z = np.polyfit(cols, c, 2)
        p = np.polyval(z, cols)
        # plt.plot(c)
        # plt.plot(p)
        # plt.show()
        # plt.close()
        max_offset = int(np.max(c))
        new_data = np.zeros((numrows+max_offset, numcols))

        for i in cols:
            # new_data[:, i] = np.roll(data[:, i], int(-c[i]))
            offset = max_offset - int(c[i])
            # print(offset, offset+numrows, data[:, i].shape)
            new_data[offset: offset + numrows, i] = self.data[:, i]
        # plt.imshow(np.log(new_data+1.0))
        plt.plot(new_data[:, 800])
        plt.show()
        plt.close()
        plt.plot(np.mean(new_data[620: 640, :], 0))
        plt.plot(np.mean(new_data[681: 703, :], 0))
        plt.show()
        plt.close()


e = Echelle()
e.loadimage('/Users/jwe/Projects/Astronomie/Spectra/spectrum7.fit')
e.artificial_flat()
e.saveimage('/Users/jwe/Projects/Astronomie/Spectra/spectrum7_flat.fit')
# e.extract_orders()
