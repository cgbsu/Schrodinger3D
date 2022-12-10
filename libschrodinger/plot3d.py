import pyqtgraph as pg
import pyqtgraph.opengl as pggl
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from libschrodinger.numerov3d import DimensionIndex
from libschrodinger.numerov3d import MeshGrid
from libschrodinger.numerov3d import WaveFunctions

class GPUPlot3D: 
    def __init__(self, application, data : np.ndarray, lower = 0, upper = 10):
        assert False, "Broken"
        self.application = application
        #self.grid = grid.toArray()
        self.data = data
        self.colors = np.empty(data.shape + (4,), dtype=np.ubyte)
        self.decibles = -10 * np.log10(self.data)
        self.max = self.decibles.max()
        self.min = self.decibles.min() * .75
        print(self.data)
        self.colors[..., 0] = self.data % 255 #self.decibles
        #np.where(
        #        (self.decibles > self.max) & (self.decibles < self.min), 
        #        0, 
        #        self.decibles
        #    )
        self.colors[..., 3] = 150#(self.colors[..., 0] * 255) % 255
        self.view = pggl.GLViewWidget()
        self.view.show()
        self.grid = pggl.GLGridItem()
        self.plot = pggl.GLVolumeItem(self.colors)
        self.axis = pggl.GLAxisItem()
        self.view.addItem(self.grid)
        self.view.addItem(self.plot)
        self.view.addItem(self.axis)



class Plot3D: 
    def __init__(
               self, 
               currentEnergyIndex : int, 
               grid : MeshGrid, 
               potential : np.ndarray, 
               waves : WaveFunctions, 
               properties : list[bool] = [True, True, True, True]
           ):
        self.figure = None
        self.currentEnergyIndex = currentEnergyIndex
        self.grid = grid
        self.potential = potential
        self.waves = waves
        self.properties = properties 
        self.plot()

    def nextEnergy(self, event):
        self.currentEnergyIndex += 1
        self.plot()

    def previousEnergy(self, event):
        if self.currentEnergyIndex > 0: 
            self.currentEnergyIndex -= 1
            self.plot()

    def plot(self):
        print("Energy Index: " + str(self.currentEnergyIndex))
        self.figure = plt.figure(0, figsize=(9, 9))
        self.nextButton = plt.Button(self.figure.add_axes([0.7, 0.05, 0.1, 0.075]), "Next")
        self.previousButton = plt.Button(self.figure.add_axes([0.81, 0.05, 0.1, 0.075]), "Previous")
        self.nextButton.on_clicked(self.nextEnergy)
        self.previousButton.on_clicked(self.previousEnergy)
        toPlot = [
                self.potential, 
                self.waves.waveFunctions[self.currentEnergyIndex], 
                self.waves.probabilities[self.currentEnergyIndex], 
                self.waves.decibleProbabilities[self.currentEnergyIndex], 
            ]
        titles = [ 
                 "Potential", 
                 "Wave Function", 
                 "Probability Distribution", 
                 "Probability Distribution (Decibles)"
             ]
        axis = []
        for ii in range(4): 
            if self.properties[ii] == True: 
                axis.append(self.figure.add_subplot(2, 2, ii + 1, projection="3d"))
                scatter = axis[-1].scatter3D(
                        self.grid.x, 
                        self.grid.y, 
                        self.grid.z, 
                        c = toPlot[ii], 
                        cmap = cm.seismic, 
                        s = 0.001, 
                        alpha = .6, 
                        antialiased = True
                    )
                axis[-1].set_title(titles[ii])
                self.figure.colorbar(scatter)
        plt.show()
    
# https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html
class IndexTracker:
    def __init__(
                self, 
                potential, 
                pointCount, 
                currentEnergy, 
                Xs, 
                sliderDimenisons = [0.0, 0.25, 0.0225, 0.63]
            ):
        self.figure = plt.figure(0, figsize=(9, 9))
        self.potential = potential
        self.ax = []
        self.titles = [ 
                 "Potential", 
                 "Wave Function", 
                 "Probability Distribution", 
                 "Probability Distribution (Decibles)"
             ]
        for ii in range(4): 
            self.ax.append(self.figure.add_subplot(2, 2, ii + 1))
            self.ax[-1].set_title(self.titles[ii])
        self.Xs = Xs
        self.currentEnergy = currentEnergy
        self.nextButton = plt.Button(self.figure.add_axes([0.7, 0.05, 0.1, 0.075]), "Next")
        self.previousButton = plt.Button(self.figure.add_axes([0.81, 0.05, 0.1, 0.075]), "Previous")
        self.nextButton.on_clicked(self.nextEnergy)
        self.previousButton.on_clicked(self.previousEnergy)
        rows, cols, self.slices = self.potential.shape
        self.ims = [
                self.ax[0].imshow(self.potential[:, :, self.slices // 2]), 
                self.ax[1].imshow(self.Xs.waveFunctions[self.currentEnergy][:, :, self.slices // 2]), 
                self.ax[2].imshow(self.Xs.probabilities[self.currentEnergy][:, :, self.slices // 2]), 
                self.ax[3].imshow(self.Xs.decibleProbabilities[self.currentEnergy][:, :, self.slices // 2])
            ]
        self.slider = plt.Slider(
            ax=self.figure.add_axes(sliderDimenisons), 
            label="z",
            valmin=0,
            valmax=pointCount,
            valstep=1, 
            valinit=self.slices // 2, 
            orientation="vertical"
        )
        self.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.slider.on_changed(self.update)
        self.update()

    def on_scroll(self, event):
        self.update()

    def nextEnergy(self, event): 
        self.currentEnergy += 1
        self.update()

    def previousEnergy(self, event): 
        if self.currentEnergy > 0:
            self.currentEnergy -= 1
            self.update()

    def update(self, sliderValue = None):
        self.figure.suptitle("Energy Index: " + str(self.currentEnergy))
        index = int(sliderValue) if sliderValue else int(self.slider.val)
        self.ims[0].set_data(self.potential[:, :, index])
        self.ims[1].set_data(self.Xs.waveFunctions[self.currentEnergy][:, :, index])
        self.ims[2].set_data(self.Xs.probabilities[self.currentEnergy][:, :, index])
        self.ims[3].set_data(self.Xs.decibleProbabilities[self.currentEnergy][:, :, index])
        for im in self.ims: 
            self.figure.colorbar(im)
        self.ax[0].set_ylabel('slice %s' % index)
        print(index)
        for im in self.ims: 
            im.axes.figure.canvas.draw()
        #self.im.axes.figure.canvas.draw_idle()

