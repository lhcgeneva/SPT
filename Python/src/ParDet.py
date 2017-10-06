from IPython.display import HTML
import numpy as np
from matplotlib.pyplot import cla, figure, Normalize, show, subplots
from matplotlib import animation, rc
from multiprocessing import Pool


class ParSim(object):

    def __init__(self, Atot=1, bc='PER', d=0.15, dt=0.05, grid_size=100, ka=1,
                 koff=0.005, kon=0.006, ratio=1, S_to_V=0.174,
                 sys_size=134.6, T=60000,):

        self.bc = bc
        self.d = d
        self.ka = ka
        self.kon = kon
        self.koff = koff
        self.ratio = ratio

        self.Atot = Atot
        self.dt = dt  # time step
        self.Ptot = ratio*Atot
        self.S_to_V = S_to_V
        self.size = grid_size  # length of grid
        self.T = T  # wall time

        self.dx = sys_size/self.size  # space step
        self.n = int(self.T/self.dt)
        self.Acyto = np.ones(self.n)
        self.Pcyto = np.ones(self.n)

    def set_init_profile(self):

        self.A = np.ones((self.size, self.n))*1.0
        self.P = np.ones((self.size, self.n))*1.0

        if self.bc == 'PER':
            quarter = int(np.round(self.size/4))
            self.A[quarter:int(np.round(self.size/2)+quarter), 0] = 0
            self.P[0:quarter, 0] = 0
            self.P[-quarter:, 0] = 0
        elif self.bc == 'NEU':
            self.A[int(np.round(self.size/2)):, 0] = 0
            self.P[0:int(np.round(self.size/2)), 0] = 0

    def show_movie(self):
        fig, ax = subplots()

        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1.5))

        line, = ax.plot([], [], lw=2)
        lineA, = ax.plot([], [], lw=2)
        lineP, = ax.plot([], [], lw=2)
        t_text = ax.text(0.1, 0.2, r't[s] = ' + str(0), fontsize=15)

        # initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            return (line,)

        # animation function. This is called sequentially
        def animate(i):
            x = np.linspace(0, 1, self.A.shape[0])
            yA = self.A[:, i]
            yP = self.P[:, i]
            lineA.set_data(x, yA)
            lineP.set_data(x, yP)
            t_text.set_text(r't[s] = ' + str(i*self.dt))
            return (line,)

        # call the animator.
        index_e = np.argmax(np.sum(self.A, 0) == 100)
        a = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=range(0, index_e, int(index_e/100)),
                                    interval=100, blit=False)
        a.save('lines.mp4')

    def simulate(self):
        if self.bc == 'PER':
            def laplacian(Z):
                l = len(Z)
                Z = np.r_[Z, Z, Z]
                Zleft = Z[0:-2]
                Zright = Z[2:]
                Zcenter = Z[1:-1]
                LAP = (Zleft + Zright - 2*Zcenter) / self.dx**2
                return LAP[l-1:2*l-1]
            self.set_init_profile()
            for i in range(self.n-1):
                deltaA = laplacian(self.A[:, i])
                deltaP = laplacian(self.P[:, i])
                Ac = self.A[:, i]
                Pc = self.P[:, i]
                self.Acyto[i] = self.Atot - self.S_to_V*np.sum(Ac)/self.size
                self.Pcyto[i] = self.Ptot - self.S_to_V*np.sum(Pc)/self.size
                self.A[:, i+1] = Ac+self.dt*(self.d*deltaA - self.koff*Ac +
                                             self.kon*self.Acyto[i] -
                                             self.ka*self.A[:, i] *
                                             self.P[:, i]**2)
                self.P[:, i+1] = Pc+self.dt*(self.d*deltaP - self.koff*Pc +
                                             self.kon*self.Pcyto[i] -
                                             self.ka*self.A[:, i]**2 *
                                             self.P[:, i])
                if sum(self.A[:, i]-self.A[:, i+1]) == 0:
                    print('steady state reached')
                    break
        elif self.bc == 'NEU':
            def laplacian(Z):
                Zleft = Z[0:-2]
                Zright = Z[2:]
                Zcenter = Z[1:-1]
                return (Zleft + Zright - 2*Zcenter) / self.dx**2
            self.set_init_profile()
            for i in range(self.n-1):
                deltaA = laplacian(self.A[:, i])
                deltaP = laplacian(self.P[:, i])
                Ac = self.A[1:-1, i]
                Pc = self.P[1:-1, i]
                self.Acyto[i] = self.Atot - self.S_to_V*np.sum(Ac)/self.size
                self.Pcyto[i] = self.Ptot - self.S_to_V*np.sum(Pc)/self.size
                self.A[1:-1, i+1] = Ac+self.dt*(self.d*deltaA - self.koff*Ac +
                                                self.kon*self.Acyto[i] -
                                                self.ka*self.A[1:-1, i] *
                                                self.P[1:-1, i]**2)
                self.P[1:-1, i+1] = Pc+self.dt*(self.d*deltaP - self.koff*Pc +
                                                self.kon*self.Pcyto[i] -
                                                self.ka*self.A[1:-1, i]**2 *
                                                self.P[1:-1, i])
                # Neumann conditions
                for Z in (self.A, self.P):
                    Z[0, i+1] = Z[1, i+1]
                    Z[-1, i+1] = Z[-2, i+1]

                if sum(self.A[:, i]-self.A[:, i+1]) == 0:
                    print('steady state reached')
                    break


class Sim_Container:

    def __init__(self, param_dict, no_workers=8):
        self.no_workers = no_workers
        self.param_dict = param_dict

    def init_simus(self):
        self.simList = []
        for k in self.param_dict['koff']:
            self.simList.append(ParSim(koff=k))

    def run_simus(self):
        # Create pool, use starmap to pass more than one parameter, do work
        pool = Pool(processes=self.no_workers)
        self.res = pool.map(sim_indiv, self.simList)


def sim_indiv(Simu):
    Simu.simulate()
    return Simu
