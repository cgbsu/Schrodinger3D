from libschrodinger.numerov3d import *
from libschrodinger.plot3d import *
from libschrodinger.potentials3d import *
import matplotlib
import pyqtgraph as pg

def main(): 
    with cp.cuda.Device(0): 
        pointCount : int = 50
        grid = makeLinspaceGrid(pointCount, 1, 3)
        potential = tunnelingCase(grid, .6, .1, 1e3)
        print("Built potential, calculating wave functions")
        waves = computeWaveFunction(potential, energyCount = 20) 
        print("Done computing wave functions, with corresponding energies, please wait for graphical output.")
        currentEnergy = 0
        application = pg.mkQApp()
        plots = GPUAcclerated3DPlotApplication(application, potential, waves)
        application.instance().exec()


if __name__ == "__main__": 
    main()

