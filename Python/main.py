import time
# from MovieTracks import ParticleFinder
from MovieTracks import DiffusionFitter

if __name__ == "__main__":
    fol = '/Users/hubatsl/Desktop/DataSantaBarbara/Aug_09_10_Test_SPT/th411_P1_40ms_100p_299gain_1678ang.nd2'
    d = DiffusionFitter(fol, 1400, parallel=True, pixelSize=0.104, timestep=0.0402,
                        saveFigs=True, showFigs=False, autoMetaDataExtract=False)
    t = []
    t0 = time.time()
    for i in range(1):
        # d.analyze()
        t.append(time.time() - t0)
    print(t)
