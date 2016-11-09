import bisect
from IPython.core.debugger import Tracer
from itertools import repeat
import numpy as np
import scipy.stats as stats
from multiprocessing import Pool
from pandas import DataFrame
from tifffile import imsave


class ImageSimulator(object):

    def __init__(self, DiffCoeffs, noPerSpecies, aboveBG=550,
                 fname='test/temp', frameInt=0.033, imageBaseValue=950,
                 no_workers=8, noiseVariance=200, pixelSize=0.124,
                 resolution=0.1, Sigma=1, timemax=6, varCoeffDet=0.1):
        # arguments
        self.Ds = DiffCoeffs / pixelSize**2
        self.noPerSpecies = noPerSpecies
        # keyword arguments
        self.aboveBG = aboveBG
        self.fname = fname
        self.frameInt = frameInt
        self.imageBaseValue = imageBaseValue  # match experimental BG values
        self.no_workers = no_workers          # no. of CPU cores to use
        self.noiseVariance = noiseVariance    # approx. noise variance in exps.
        self.pixelSize = pixelSize            # in microns
        self.resolution = resolution          # subpixel-resolution
        self.Sigma = Sigma                    # spread of PSF in pixels
        self.varCoeffDet = varCoeffDet        # coefficient of variation for D
        self.timemax = timemax                # movie time
        # set problem specific parameters that need not normally be changed
        self.a = 27/self.pixelSize  # half of long axis of the embryo in µm
        self.b = 30/self.pixelSize  # short axis of embryo in µm
        self.N = sum(self.noPerSpecies)  # number of molecules
        self.D = np.repeat(self.Ds, self.noPerSpecies)  # mean D repeated
        self.var = self.varCoeffDet * self.D  # variance for each particle
        # different D for every particle
        self.d = np.random.normal(self.D, self.var, (1, self.N))

        # allocate memory for particles
        self.framemax = int(np.floor(self.timemax/self.frameInt))
        self.allx = np.zeros((self.framemax, self.N))
        self.ally = np.zeros((self.framemax, self.N))
        # get initial particle positions
        self.allx[0, :] = self.a*(np.random.rand(1, self.N) -
                                  np.ones((1, self.N))*0.5)
        self.ally[0, :] = self.b*(np.random.rand(1, self.N) -
                                  np.ones((1, self.N))*0.5)
        # create individual tracks for each particle
        for part in range(self.N):
            for framenum in range(1, self.framemax):
                self.allx[framenum, part] = (self.allx[framenum-1, part] +
                                             np.sqrt(abs(2*self.frameInt *
                                                     self.d[0, part])) *
                                             np.random.randn(1))
                self.ally[framenum, part] = (self.ally[framenum-1, part] +
                                             np.sqrt(abs(2*self.frameInt *
                                                     self.d[0, part])) *
                                             np.random.randn(1))
        # make sure nothing falls off the edge
        self.allx = self.allx - np.amin(self.allx)
        self.ally = self.ally - np.amin(self.ally)
        # get size of movie including t direction, initialize noise
        self.size_synthetic_movie = [int(np.ceil(np.amax(self.allx)) + 6),
                                     int(np.ceil(np.amax(self.ally)) + 6),
                                     self.framemax]
        self.allx = self.allx + 3
        self.ally = self.ally + 3
        self.imMat = abs(self.noiseVariance *
                         np.random.normal(size=self.size_synthetic_movie) +
                         self.imageBaseValue)

    def create_images(self):
        '''
        Convolves each particle with a PSF to simulate imaging,
        projects onto pixel grid to simulate camera
        '''
        # use parallel pool to get convolution done
        pool = Pool(processes=self.no_workers)
        res = pool.starmap(
                sum_gaussians,
                zip(repeat(self.size_synthetic_movie[0:2]),
                    repeat(self.resolution),
                    [np.c_[self.allx[i, :], self.ally[i, :]]
                        for i in range(self.framemax)],
                    repeat(self.Sigma)))
        pool.close()
        pool.join()
        # add to noisy images
        for i, im in enumerate(res):
            self.imMat[:, :, i] = (self.imMat[:, :, i] +
                                   self.aboveBG/np.mean(im[im > 0])*im)

    def write_images(self):
        for i in range(self.imMat.shape[2]):
            imsave(self.fname+str(i)+'.tif', np.uint16(self.imMat[:, :, i]))

    def write_log(self):
        columns = ['DiffCoeffs', 'noPerSpecies', 'aboveBG=550', 'fname',
                   'frameInt', 'imageBaseValue', 'no_workers', 'noiseVariance',
                   'pixelSize', 'resolution', 'Sigma', 'timemax',
                   'varCoeffDet']
        values = [str(self.Ds*self.pixelSize**2), str(self.noPerSpecies),
                  self.aboveBG, self.fname, self.frameInt, self.imageBaseValue,
                  self.no_workers, self.noiseVariance, self.pixelSize,
                  self.resolution, self.Sigma, self.timemax, self.varCoeffDet]
        self.dfInput = DataFrame(columns=columns)
        self.dfInput.loc[0] = values
        self.dfInput.to_csv(self.fname + '_in.csv')


def sum_gaussians(gridsize, resolution, partPos, Sigma):
    '''
    Creates image convolved with an artificial point spread function.
    PSF has breadth Sigma, within each pixel of image gridsize
    '''

    # Initialize arrays
    sz = np.shape(partPos)
    numSteps = 4*Sigma/resolution+1   # 1-D interpolation length
    interpLen = int(numSteps**2)      # len of interpolation around point
    arraysize = int(interpLen*sz[0])  # len of containers for x, y, p
    xall = np.zeros(arraysize)  # container for x
    yall = np.zeros(arraysize)  # container for y
    pall = np.zeros(arraysize)  # container for probabilities
    imMat = np.zeros(gridsize)  # image matrix to project particles on

    # For each particle, create a meshgrid of size 2*Sigma around that
    # particle, link them all together and evaluate gaussian of
    # spread Sigma and mean partPos at each point of the meshgrid.
    for i in range(sz[0]):
        X = np.linspace(partPos[i, 0]-2*Sigma, partPos[i, 0]+2*Sigma, numSteps)
        Y = np.linspace(partPos[i, 1]-2*Sigma, partPos[i, 1]+2*Sigma, numSteps)
        x, y = np.meshgrid(X, Y)
        xall[i*interpLen:(i+1)*interpLen] = x.flatten(order='F')
        yall[i*interpLen:(i+1)*interpLen] = y.flatten(order='F')
        pall[i*interpLen:(i+1)*interpLen] = stats.multivariate_normal.pdf(
                                        np.c_[x.flatten('F'), y.flatten('F')],
                                        partPos[i, :], Sigma)
    # Sort by ascending x-values
    xSortInds = np.argsort(xall, kind='mergesort')
    xsorted = xall[xSortInds]
    ysorted = yall[xSortInds]
    psorted = pall[xSortInds]

    # Project particles onto image.
    # Iterate over x dimension of image
    for i in range(gridsize[0]):
        temp_min = bisect.bisect_left(xsorted, i)-1
        temp_max = bisect.bisect_right(xsorted, i+1)
        for j in range(gridsize[1]):
            if not (temp_min == -1 or temp_max == -1):
                ytemp = ysorted[temp_min:temp_max+1]
                ptemp = psorted[temp_min:temp_max+1]
                # Sum all values from pall that have an x and y value within
                # the current pixel (i, j).
                imMat[i, j] = np.sum(ptemp[np.logical_and(ytemp > j,
                                           ytemp <= j+1)])
    return imMat
