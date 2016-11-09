# SPT (Membrane **S**ingle **P**article **T**racking in C. elegans)
## Python
Code is divided into src and TestNbs (jupyter test notebooks). The latter are used to compare different algorithms and test the code.

*Class MovieTracks built on the [trackpy project](https://github.com/soft-matter/trackpy) to track membrane bound fluorescent proteins in live imaging movies.
*Class SimDiffusion to simulate particle movement movies in 2D. Creates a series of images with diffusive particles overlayed on top of noisy background. Spots are blurred by Gaussian filter to simulate a microscope point spread function. This class is used to create images to benchmark algorithms relying on detection and tracking of bright fluorescent spots.

---

## Matlab
Set of scripts to achieve functionality similar to Python code explained above. This uses the [Crocker Grier](http://crocker.seas.upenn.edu/CrockerGrier1996b.pdf) implementation used in 
[Robin et al. (2014)](http://www.nature.com/nmeth/journal/v11/n6/full/nmeth.2928.html) originally written by written by the [Kilfoil lab](http://people.umass.edu/kilfoil/downloads.html).
