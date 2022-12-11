from libschrodinger.numerov3d import *
from libschrodinger.plot3d import *
import matplotlib
import pyqtgraph as pg

def main(): 
    with cp.cuda.Device(0): 
        pointCount : int = 50
        grid = makeLinspaceGrid(pointCount, 1, 3)
        potential = 0 * grid.x
        waves = computeWaveFunction(potential, energyCount = 100) 
        currentEnergy = 0
        application = pg.mkQApp()
        plots = GPUAcclerated3DPlotApplication(application, potential, waves)
        application.instance().exec()


if __name__ == "__main__": 
    main()

