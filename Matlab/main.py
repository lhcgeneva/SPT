import time
from MovieTracks import ParticleFinder as PF

if __name__ == "__main__":
    fol = '/Users/hubatsl/Desktop/SPT/Us/Diffusion/PAR6/16_04_10_TH411_M9/fov5/'
    m = PF(fol, 600, 0.033, parallel=True, showFigs=True, saveFigs=True)
# m = MT('/Users/hubatsl/Desktop/33ms_100p_2.tif', 1000, 0.033, parallel=True)
    t = []
    for i in range(1):
        t0 = time.time()
        # m.find_feats()
        m.analyze()
        t.append(time.time() - t0)
    print(t)
