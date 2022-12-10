from libschrodinger.numerov3d import *
from libschrodinger.plot3d import *

def hydrogenAtom(grid : MeshGrid, centerX, centerY, centerZ, bottom, potential) -> np.ndarray: 
    return potential / (np.sqrt(
            (grid.x - centerX) ** 2 \
            + (grid.y - centerY) ** 2 \
            + (grid.z - centerZ) ** 2 \
            + bottom ** 2 \
        ))

def main(): 
    matplotlib.use('TkAgg')
    with cp.cuda.Device(0): 
        pointCount : int = 150
        grid = makeLinspaceGrid(pointCount, 1, 3)
        potential = hydrogenAtom(grid, .5, .5, .5, .1, 1)
        print("Built potential, calculating wave functions")
        waves = computeWaveFunction(potential)
        print("Done computing wave functions, with corresponding energies, please wait for graphical output.")
        plot = Plot3D(0, grid, potential, waves)
        #plot = IndexTracker(potential, pointCount, 0, waves)
        print("Done plotting!")
        #plt.show()


if __name__ == "__main__": 
    main()

