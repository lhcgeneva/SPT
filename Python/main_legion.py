import sys
import numpy
from MovieTracks import ParameterSampler

#Get noise value for each run from command line argument
arg = sys.argv[1]
noise = numpy.array([float(arg)/100])
#For each noise value run the following param combinations
offRate = numpy.arange(0.0005, 0.02, 0.0005)
kOn = numpy.arange(50, 1000, 50)
kPh = numpy.arange(0.0005, 0.02, 0.0005)
#Do the sampling
ParameterSampler(offRate, kPh, kOn, noise, arg)
