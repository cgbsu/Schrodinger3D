from libschrodinger.numerov3d import *
from libschrodinger.plot3d import *
import matplotlib
import pyqtgraph as pg

def hydrogenAtom(grid : MeshGrid, centerX, centerY, centerZ, bottom, potential) -> np.ndarray: 
    return potential / np.sqrt(
            (grid.x - centerX) ** 2 \
            + (grid.y - centerY) ** 2 \
            + (grid.z - centerZ) ** 2 \
            + bottom ** 2 \
        )

def main(): 
    with cp.cuda.Device(0): 
        pointCount : int = 50
        grid = makeLinspaceGrid(pointCount, 1, 3)
        potential = 0 * grid.x#hydrogenAtom(grid, .5, .5, .5, 1e-3, 1)
        waves = computeWaveFunction(potential)
        currentEnergy = 0
        application = pg.mkQApp()
        plotPotential = GPUPlot3D(application, potential)
        plotWaveFunction = GPUPlot3D(application, waves.waveFunctions[currentEnergy])
        plotProbabilities = GPUPlot3D(application, waves.probabilities[currentEnergy])
        plotDecibleProbabilities = GPUPlot3D(application, waves.decibleProbabilities[currentEnergy])
        application.instance().exec()


if __name__ == "__main__": 
    main()

