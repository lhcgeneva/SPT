from IPython.core.debugger import Tracer
from IPython.display import HTML
from numpy import argmax, ceil, linspace, ones, r_, round, sum
from matplotlib.pyplot import cla, figure, Normalize, show, subplots
from matplotlib import animation, rc
from multiprocessing import Pool


class ParSim(object):

    def __init__(self, Atot=1, bc='PER', d=0.15, dt=0.05, grid_size=100, ka=1,
                 koff=0.005, kon=0.006, ratio=1, save_nth=100, StoV=0.174,
                 sys_size=134.6, T=60000):

        self.Atot = Atot
        self.bc = bc
        self.d = d
        self.dt = dt  # time step
        self.ka = ka
        self.kon = kon
        self.koff = koff
        self.ratio = ratio
        self.Ptot = ratio*Atot
        self.StoV = StoV
        self.save_nth = save_nth
        self.size = grid_size  # length of grid
        self.T = T  # wall time

        self.dx = sys_size/self.size  # space step
        self.n = int(self.T/self.dt)
        self.Acy = ones(int(ceil(self.n/self.save_nth)))
        self.Pcy = ones(int(ceil(self.n/self.save_nth)))

    def set_init_profile(self):

        self.A = ones((self.size, int(ceil(self.n/self.save_nth))))*1.0
        self.P = ones((self.size, int(ceil(self.n/self.save_nth))))*1.0

        if self.bc == 'PER':
            quarter = int(round(self.size/4))
            self.A[quarter:int(round(self.size/2)+quarter), 0] = 0
            self.P[0:quarter, 0] = 0
            self.P[-quarter:, 0] = 0
        elif self.bc == 'NEU':
            self.A[int(round(self.size/2)):, 0] = 0
            self.P[0:int(round(self.size/2)), 0] = 0

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
            x = linspace(0, 1, self.A.shape[0])
            yA = self.A[:, i]
            yP = self.P[:, i]
            lineA.set_data(x, yA)
            lineP.set_data(x, yP)
            t_text.set_text(r't[s] = ' + str(i*self.dt*self.save_nth))
            return (line,)

        # call the animator.
        index_e = argmax(sum(self.A, 0) == self.size)
        a = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=range(0, index_e, int(index_e/100)),
                                    interval=100, blit=False)
        a.save('lines.mp4')

    def simulate(self):
        if self.bc == 'PER':
            def laplacian(Z):
                l = len(Z)
                Z = r_[Z, Z, Z]
                Zleft = Z[0:-2]
                Zright = Z[2:]
                Zcenter = Z[1:-1]
                LAP = (Zleft + Zright - 2*Zcenter) / self.dx**2
                return LAP[l-1:2*l-1]
            self.set_init_profile()
            An = self.A[:, 0]
            Pn = self.P[:, 0]
            for i in range(self.n-1):
                Ac = An
                Pc = Pn
                deltaA = laplacian(Ac)
                deltaP = laplacian(Pc)
                Acy = self.Atot - self.StoV*sum(Ac)/self.size
                Pcy = self.Ptot - self.StoV*sum(Pc)/self.size
                # Tracer()()
                An = Ac+self.dt*(self.d*deltaA - self.koff*Ac +
                                 self.kon*Acy -
                                 self.ka*Ac*Pc**2)
                Pn = Pc+self.dt*(self.d*deltaP - self.koff*Pc +
                                 self.kon*Pcy -
                                 self.ka*Ac**2*Pc)

                # Save every save_nth frame, check whether steady state reached
                if i % self.save_nth == 0:
                    j = int(i/self.save_nth)
                    self.A[:, j] = Ac
                    self.P[:, j] = Pc
                    self.Acy[j] = Acy
                    self.Acy[j] = Pcy
                    if sum(Ac-An) == 0:
                        # Append the next frame (An), to show that steady
                        # state was reached
                        self.A[:, j+1] = An
                        self.P[:, j+1] = Pn
                        self.Acy[j+1] = self.Atot - self.StoV*sum(An)/self.size
                        self.Acy[j+1] = self.Ptot - self.StoV*sum(Pn)/self.size
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
                self.Acy[i] = self.Atot - self.StoV*sum(Ac)/self.size
                self.Pcy[i] = self.Ptot - self.StoV*sum(Pc)/self.size
                self.A[1:-1, i+1] = Ac+self.dt*(self.d*deltaA - self.koff*Ac +
                                                self.kon*self.Acy[i] -
                                                self.ka*self.A[1:-1, i] *
                                                self.P[1:-1, i]**2)
                self.P[1:-1, i+1] = Pc+self.dt*(self.d*deltaP - self.koff*Pc +
                                                self.kon*self.Pcy[i] -
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
