#!/usr/bin/env python
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
from scipy import interpolate
from stackoverflow import find_outlier_pixels

indir = "/work2/jwe/Projects/Astro/spectra/"
outdir = "/work2/jwe/Projects/Astro/spectra/"

class Echelle(object):
    def __init__(self):
        self.data = None
        self.dark = None

    def loadimage(self, filename):
        hdu = fits.open(filename)
        self.data = hdu[0].data

    def saveimage(self, filename):
        hdu = fits.PrimaryHDU(self.data)
        hdu.writeto(filename, overwrite=True)

    def subtract_dark(self, filename):
        hdu = fits.open(filename)
        self.dark = hdu[0].data
        hot_pixels, self.dark = find_outlier_pixels(self.dark, worry_about_edges=True)
        print(hot_pixels)
        self.data = self.data - self.dark
        i = np.where(self.data < 0)
        self.data[i] = 0.0
        i = np.where(self.data > 55000)
        self.data[i] = 0.0

    def normalize(self):
        # normalization
        self.data = 1.0 * self.data - np.min(self.data)
        self.data /= np.max(self.data)

    def remove_hot_pixels(self):
        hot_pixels, self.data = find_outlier_pixels(self.data, worry_about_edges=True)
        print(hot_pixels)

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

        X = X.flatten()
        Y = Y.flatten()

        A = np.array([X * 0 + 1, X, Y, X ** 2, Y ** 2, X * Y]).T
        B = Z.flatten()

        coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=1e-5)
        X, Y = np.meshgrid(x, y, copy=False)
        flat = coeff[0] + coeff[1]*X + coeff[2]*Y + coeff[3]*X**2 + coeff[4]*Y**2 + coeff[5]*X*Y

        # normalize the flat since we get negative values
        flat -= np.min(flat)
        flat /= np.max(flat)
        flat += 0.01
        hdu = fits.PrimaryHDU(flat)
        hdu.writeto(outdir+'artificial_flat.fit', overwrite=True)

        self.data /= flat
        self.data -= np.nanmin(self.data)
        self.data /= np.nanmax(self.data)

    def extract_orders(self):
        numrows, numcols = self.data.shape
        # cols = 300
        c = np.zeros(numcols)
        cols = np.arange(numcols)
        window = 25
        reference = self.data[:, 0]
        plt.figure(figsize=[12/2.54,16/2.54])
        plt.title('order extraction')
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
            if not i % 10:
                plt.plot(np.arange(-window, window), cv[mid - window + prediction: mid + window + prediction])
        plt.grid()
        plt.savefig(outdir + 'extract_orders.pdf')
        plt.close()

        plt.figure(figsize=[16/2.54,12/2.54])
        plt.title('order prediction')
        plt.plot(c, 'ro', fillstyle='none', alpha=0.5)
        plt.plot(prediction, 'g+', alpha=0.5)
        plt.xlabel('CCD column')
        plt.savefig(outdir + 'order_prediction.pdf')

        plt.close()
        max_offset = int(np.max(c))
        new_data = np.zeros((numrows+max_offset, numcols))

        for i in cols:
            # new_data[:, i] = np.roll(data[:, i], int(-c[i]))
            offset = max_offset - int(c[i])
            # print(offset, offset+numrows, data[:, i].shape)
            new_data[offset: offset + numrows, i] = self.data[:, i]
        plt.figure(figsize=[16/2.54,12/2.54])
        plt.title('order sample')
        plt.plot(np.mean(new_data[620: 640, :], 0))
        plt.plot(np.mean(new_data[681: 703, :], 0))
        plt.xlabel('CCD column')
        plt.ylabel('intensity')
        plt.savefig(outdir + 'order_sample.pdf')
        plt.close()
        self.data = new_data
        self.normalize()
        plt.imsave(outdir + 'orders_rectified.png', new_data, vmin=0.0, vmax=np.median(new_data) * 10.0)
        hdu = fits.PrimaryHDU(self.data)
        hdu.writeto(outdir + 'orders_rectified.fit', overwrite=True)

    def subtract_background(self):
        numrows, numcols = self.data.shape
        background = np.zeros_like(self.data)
        for col in np.arange(numcols):
            row = self.data[:, col]
            i = np.where((row < np.mean(row)) & (row > 0.0))
            y = np.arange(numrows)
            z = np.polyfit(y[i], row[i], 5)
            p = np.polyval(z, y)
            background[:, col] = p
            plt.plot(row)
            plt.plot(y[i], row[i],',')
        hdu = fits.PrimaryHDU(background)
        hdu.writeto(outdir+'artificial_background.fit', overwrite=True)

        plt.plot(y, p)
        plt.show()
        plt.close()

    def continuum_correction(self, wavelengths, intensities):
        reference_wavelengths, reference_intensities = self.load_reference()
        interref = interpolate.interp1d(reference_wavelengths, reference_intensities, bounds_error=False)
        reference_wavelengths = wavelengths
        reference_intensities = interref(wavelengths)

        i = np.where(reference_intensities > 0.99)
        reference_wavelengths = reference_wavelengths[i]
        reference_intensities = reference_intensities[i]
        continuum_wavelengths = wavelengths[i]
        continuum_intensities = intensities[i]
        p = np.polyfit(continuum_wavelengths, continuum_intensities, 5)
        continuum_correction = np.polyval(p, wavelengths)
        plt.plot(continuum_wavelengths, continuum_intensities, 'o')
        plt.plot(reference_wavelengths, reference_intensities, 'x')
        plt.plot(wavelengths, continuum_correction)
        plt.grid()
        plt.xlabel('wavelength [nm]')
        plt.ylabel('flux')
        plt.savefig(outdir + 'continuum_fit.pdf')
        plt.show()
        plt.close()
        return intensities - continuum_correction + 1.0

    def extract_individual_order(self):
        order = self.data[1068: 1109, :]
        order /= np.percentile(order, 75)

        columns = order.shape[1]

        def box(x, *p):
            # stackoverflow 46624376
            height, center, width = p
            return height*(center-width/2 < x)*(x < center+width/2)

        m = np.zeros(columns)
        s = np.zeros(columns)

        for column in range(columns):
            x = np.arange(order.shape[0])
            y = order[:, column]
            ymin = np.percentile(y, 25)
            y -= ymin
            per66 = np.percentile(y, 66)
            i = np.where(y > per66)
            m[column] = np.mean(y[i])
            s[column] = np.std(y[i])
            # WARNING: hardcoded center limit!
            # res = differential_evolution(lambda p: np.sum((box(x, *p) - y)**2), [[0, 1], [0, 40], [1, 20]])
            # plt.plot(x, y, '.')

            # plt.step(x, box(x, *res.x), where='mid')
            # plt.grid()
            # plt.show()

        reference_wavelengths, reference_intensities = self.load_reference()
        spectrum_x = np.arange(columns)
        spectrum_intensities = m
        savearray = np.c_[spectrum_x, spectrum_intensities]
        np.savetxt(outdir+'order_spectrum.txt', savearray, fmt='%04d %.6f')
        def minfun(par):
            # reference_x = np.arange(len(reference_wavelengths))
            test_wavelengths = np.polyval(par, spectrum_x)
            f0 = interpolate.interp1d(reference_wavelengths, reference_intensities, bounds_error=False)
            f1 = interpolate.interp1d(test_wavelengths, spectrum_intensities, bounds_error=False)

            newcompspec = f1(test_wavelengths)
            newrefspec = f0(test_wavelengths)
            return np.std(newcompspec-newrefspec)
        x0 = self._polysolve()

        from scipy.optimize import minimize
        res = minimize(minfun, x0, method='Nelder-Mead')
        if res.success:
            x0 = res.x
            print("%.1e %.4f %.3f" % (x0[0], x0[1], x0[2]))
        else:
            print(res)
        spectrum_wavelengths = np.polyval(x0, spectrum_x)



        savearray = np.c_[spectrum_wavelengths, spectrum_x, spectrum_intensities]
        np.savetxt(outdir + 'order_spectrum.txt', savearray, fmt='%.6f %d %.6f', header="#lambda x int")
        fig, ax = plt.subplots()
        ax.plot(spectrum_wavelengths, spectrum_intensities, 'black')
        #ax.fill_between(x1, m-s, m+s, facecolor='gray', alpha=0.5)
        ax.plot(reference_wavelengths, reference_intensities, 'green', alpha = 0.5)
        ax.set_xlim(np.min(spectrum_wavelengths), np.max(spectrum_wavelengths))
        plt.grid()
        plt.xlabel('wavelength [nm]')
        plt.ylabel('flux')
        plt.savefig(outdir + 'order_fitting.pdf')
        plt.close()
        spectrum_intensities = self.continuum_correction(spectrum_wavelengths, spectrum_intensities)

        fig, ax = plt.subplots()
        ax.plot(spectrum_wavelengths, spectrum_intensities, 'black')
        # ax.fill_between(x1, m-s, m+s, facecolor='gray', alpha=0.5)
        ax.plot(reference_wavelengths, reference_intensities, 'green', alpha=0.5)
        ax.set_xlim(np.min(spectrum_wavelengths), np.max(spectrum_wavelengths))
        plt.grid()
        plt.xlabel('wavelength [nm]')
        plt.ylabel('flux')
        plt.savefig(outdir + 'order_continuum.pdf')
        plt.close()

    def load_reference(self):
        """
        source data from http://bass2000.obspm.fr/solar_spect.php
        :return: wavelength in nanometers intensity scaled to 1
        """
        wavelengths, intensities = np.genfromtxt(indir + 'spectrum.dat',unpack = True, usecols=(0,1))
        wavelengths /= 10.0
        new_intensities = interpolate.interp1d(wavelengths, intensities, bounds_error=False, kind='slinear')
        new_wavelengths = np.arange(wavelengths[0], wavelengths[-1], 0.01)
        return new_wavelengths, new_intensities(new_wavelengths)

    def _polysolve(self):
        # x = np.array([311.0, 323.0, 400.0, 548.0, 788.0, 794.7])
        # w = np.array([653.2, 653.4, 654.6, 656.3, 659.3, 659.4])


        x, w = np.genfromtxt(indir + 'order_reference.txt',unpack = True, usecols=(0,1))

        p = np.polyfit(x, w, 2)
        plt.plot(x, w, '+')
        for xi,wi in zip(x,w):
            plt.text(xi, wi, '%.1f' % wi)
        px = np.linspace(0, 1600, 100)
        plt.plot(px, np.polyval(p, px))
        plt.show()
        print(p)
        return p

e = Echelle()
e.loadimage(indir + 'spectrum_sun_005.fit')
# e.subtract_dark(indir + 'dark10s.fit')
e.saveimage(outdir + 'spectrum7_dark.fit')

e.normalize()
e.artificial_flat()
e.saveimage(outdir+'spectrum7_flat.fit')
e.extract_orders()
e.extract_individual_order()
