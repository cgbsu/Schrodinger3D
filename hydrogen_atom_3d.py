from libschrodinger.numerov3d import *
import matplotlib
from matplotlib import cm

def hydrogenAtom(grid : MeshGrid, centerX, centerY, centerZ, bottom, potential) -> np.ndarray: 
    return potential / np.sqrt(
            (grid.x - centerX) ** 2 \
            + (grid.y - centerY) ** 2 \
            + (grid.z - centerZ) ** 2 \
            + bottom ** 2 \
        )

def main(): 
    matplotlib.use('TkAgg')
    with cp.cuda.Device(0): 
        pointCount : int = 50
        grid = makeLinspaceGrid(pointCount, 1, 3)
        potential = hydrogenAtom(grid, .5, .5, .5, 1e-3, 1)
        figure = plt.figure(0, figsize=(9, 9))
        potentialAxis = figure.add_subplot(2, 2, 1, projection="3d")
        potentialAxis.scatter3D(
                grid.x, 
                grid.y, 
                grid.z, 
                c = potential, 
                cmap = cm.seismic, 
                s = 0.001, 
                alpha = .6, 
                antialiased = True
            )
        potentialAxis.set_title("Potential")
        waves = computeWaveFunction(potential)
        currentEnergy = 0
        waveAxis = figure.add_subplot(2, 2, 2, projection="3d")
        waveAxis.scatter3D(
                grid.x, 
                grid.y, 
                grid.z, 
                c = waves.waveFunctions[currentEnergy], 
                cmap = cm.seismic, 
                s = 0.001, 
                alpha = .6, 
                antialiased = True
            )
        waveAxis.set_title("Wave Function")
        probabilityAxis = figure.add_subplot(2, 2, 3, projection="3d")
        probabilityAxis.scatter3D(
                grid.x, 
                grid.y, 
                grid.z, 
                c = waves.probabilities[currentEnergy], 
                cmap = cm.seismic, 
                s = 0.001, 
                alpha = .6, 
                antialiased = True
            )
        probabilityAxis.set_title("Probability Distribution")
        decibleProbabilityAxis = figure.add_subplot(2, 2, 4, projection="3d")
        decibleProbabilityAxis.scatter3D(
                grid.x, 
                grid.y, 
                grid.z, 
                c = waves.decibleProbabilities[currentEnergy], 
                cmap = cm.seismic, 
                s = 0.001, 
                alpha = .6, 
                antialiased = True
            )
        decibleProbabilityAxis.set_title("Probability Distribution (Decibles)")
        print("Done plotting")
        plt.show()


if __name__ == "__main__": 
    main()

