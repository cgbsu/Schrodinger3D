from libschrodinger.numerov3d import *
from libschrodinger.plot3d import *
from libschrodinger.potentials3d import *

def main(): 
    matplotlib.use('TkAgg')
    with cp.cuda.Device(0): 
        pointCount : int = 50
        grid = makeLinspaceGrid(pointCount, 1, 3)
        potential = tunnelingCase(grid, .7, .5, .1, .5, 1)
        print("Built potential, calculating wave functions")
        waves = computeWaveFunction(potential)
        print("Done computing wave functions, with corresponding energies, please wait for graphical output.")
        #plot = Plot3D(0, grid, potential, waves)
        plot = IndexTracker(potential, pointCount, 0, waves)
        print("Done plotting!")
        plt.show()


if __name__ == "__main__": 
    main()

