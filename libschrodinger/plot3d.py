import sys
import pyqtgraph as pg
import pyqtgraph.opengl as pggl
from pyqtgraph.Qt import QtCore, QtGui
from PyQt6 import QtWidgets
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from libschrodinger.numerov3d import DimensionIndex
from libschrodinger.numerov3d import MeshGrid
from libschrodinger.numerov3d import WaveFunctions

MAXIMUM_32_BITS = 2 ** 32
MAXIMUM_24_BITS = 2 ** 24
MAXIMUM_16_BITS = 2 ** 16
MAXIMUM_8_BITS = 2 ** 8
COLOR_COMPONENT_MAGNITUDE = 255 ** 2
FLOAT_32_EPSILON = np.finfo(np.float32).eps

def normalizeData(data : np.ndarray, threshold = FLOAT_32_EPSILON, divideOut = False, zeroOut = False): 
    maximum = data.max()
    normalized = data
    if maximum < 0:
        maximum = np.abs(maximum)
        normalized = data + (2 * maximum)
    if maximum == 0:
        minimum = data.min()
        if minimum != 0: 
            maximum = np.abs(minimum)
        else: 
            return normalized
    normalized = normalized / maximum
    if divideOut == True: 
        checkMinimum = normalized.min()
        if checkMinimum <= threshold: 
            normalized = normalized / checkMinimum
    elif zeroOut == True: 
        normalized = np.where(normalized < threshold, 0, normalized)
    return normalized

def normalizeTo4x8BitScaledColor(normalizedData : np.ndarray, alpha) -> np.ndarray: 
    normalizedMinimum = normalizedData.min()
    normalizedMaximum = normalizedData.max()
    unsigned32 = np.uint32(np.round(normalizedData * MAXIMUM_24_BITS))
    output = np.zeros(normalizedData.shape + (4, ), dtype = np.ubyte)
    ratios = [
            1, 
            MAXIMUM_16_BITS / MAXIMUM_24_BITS, 
            MAXIMUM_8_BITS / MAXIMUM_16_BITS
        ] # Each color component will be descendingly scaled by one of these values as smaller parts of the number
    if sys.byteorder == "little": 
        ratios = list(reversed(ratios))
    for ii in range(3): 
        output[..., ii] = np.ubyte((unsigned32 >> (ii * 8)) * ratios[ii])
    output[..., 3] = alpha
    return output


def normalizeTo4x8BitColor(normalizedData : np.ndarray, alpha) -> np.ndarray: 
    normalizedMinimum = normalizedData.min()
    normalizedMaximum = normalizedData.max()
    #assert normalizedMaximum <= 1 and normalizedMinimum >= 0 \
    #        "Normalized data must be between 0 and one 1"
    unsigned32 = np.uint32(np.round(normalizedData * MAXIMUM_24_BITS))
    output = np.zeros(normalizedData.shape + (4, ), dtype = np.ubyte)
    for ii in range(3): 
        output[..., ii] = np.ubyte(unsigned32 >> (ii * 8))
    output[..., 3] = alpha
    return output

def normalizeTo4x8Bits(normalizedData : np.ndarray) -> np.ndarray: 
    normalizedMinimum = normalizedData.min()
    normalizedMaximum = normalizedData.max()
    #assert normalizedMaximum <= 1 and normalizedMinimum >= 0 \
    #        "Normalized data must be between 0 and one 1"
    unsigned32 = np.uint32(np.round(normalizedData * MAXIMUM_32_BITS))
    output = np.zeros(normalizedData.shape + (4, ), dtype = np.ubyte)
    for ii in range(4): 
        output[..., ii] = np.ubyte(unsigned32 >> (ii * 8))
    return output

def normalizeTo4x8BitsStaticAlpha(alpha, normalizedData : np.ndarray) -> np.ndarray: 
    output = normalizeTo4x8Bits(normalizedData)
    x = output[..., 0]
    output[..., 0] = output[..., 1]
    output[..., 1] = output[..., 2]
    output[..., 2] = output[..., 3]
    output[..., 3] = x
    output[..., 0] = 0
    return output


class GPUPlot3D: 
    def __init__(self, application, data : np.ndarray, noiseLevel = FLOAT_32_EPSILON, alpha = 50, position = None):
        self.application = application
        self.view = pggl.GLViewWidget()
        self.grid = pggl.GLGridItem()
        self.plot = None
        self.newVolumePlot(data, noiseLevel, alpha)
        self.axis = pggl.GLAxisItem()
        #self.axisY = pggl.GLAxisItem()
        #self.axisZ = pggl.GLAxisItem()
        self.view.addItem(self.grid)
        self.view.addItem(self.axis)
        #self.view.addItem(self.axisY)
        #self.view.addItem(self.axisZ)
        self.centeredCoordinates = np.array([
                self.pointCount / 2,
                self.pointCount / 2, 
                self.pointCount / 2
            ])
        self.position = position - self.centeredCoordinates \
                if position else self.centeredCoordinates
        self.grid.translate(*tuple(self.position))
        #self.view.show()
        self.view.pan(*self.centeredCoordinates)
        self.view.opts["distance"] = 3 * self.pointCount
        #self.view.resize(800, 600)
    def newVolumePlot(self, data, noiseLevel = FLOAT_32_EPSILON, alpha = 50): 
        if self.plot: 
            self.view.removeItem(self.plot)
        self.normalizedData = normalizeData(data, noiseLevel) 
        self.normalizedData = np.where(
                self.normalizedData < noiseLevel, 
                0, 
                self.normalizedData
            )
        self.pointCount = data.shape[0] 
        self.colors = normalizeTo4x8BitScaledColor(self.normalizedData, alpha)
        self.plot = pggl.GLVolumeItem(self.colors)
        self.view.addItem(self.plot)


class GPUAcclerated3DPlotApplication: 
    Q_LABEL_STYLE_SHEET = ""
    "background-color: #262626;"
    "color: #FFFFFF;"
    "font-family: Titillium;"
    "font-size: 18px;"
    def __init__(self, application, potential, waves, currentEnergyIndex = 0): 
        self.application = application
        self.waves = waves
        self.potential = potential
        self.mainWidget = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.mainWidget.setLayout(self.layout)
        self.currentEnergyIndex = currentEnergyIndex
        self.plotPotential = GPUPlot3D(self.application, self.potential)
        self.plotWaveFunction = GPUPlot3D(self.application, self.waves.waveFunctions[currentEnergyIndex])
        self.plotProbabilities = GPUPlot3D(self.application, self.waves.probabilities[currentEnergyIndex])
        self.plotDecibleProbabilities = GPUPlot3D(self.application, self.waves.decibleProbabilities[currentEnergyIndex])
        self.energyIndexLabel = QtWidgets.QLabel("Temp")
        self.energyIndexLabel.setFixedHeight(10)
        self.energyValueLabel = QtWidgets.QLabel("Temp")
        self.energyValueLabel.setFixedHeight(10)
        self.layout.addWidget(self.energyIndexLabel, 0, 1)
        self.layout.addWidget(self.energyValueLabel, 0, 2)
        self.setLabels()
        self.plotLabels = [
                [QtWidgets.QLabel("Probabilities "), QtWidgets.QLabel("Wave Function")], 
                [QtWidgets.QLabel("Potential"), QtWidgets.QLabel("Decible Probabilities")]
            ]
        for row in self.plotLabels: 
            for label in row: 
                label.setFixedHeight(10)
        self.layout.addWidget(self.plotLabels[0][0], 1, 1)
        self.layout.addWidget(self.plotProbabilities.view, 2, 1)
        self.layout.addWidget(self.plotLabels[0][1], 1, 2)
        self.layout.addWidget(self.plotWaveFunction.view, 2, 2)
        self.layout.addWidget(self.plotLabels[1][0], 3, 1)
        self.layout.addWidget(self.plotPotential.view, 4, 1)
        self.layout.addWidget(self.plotLabels[1][1], 3, 2)
        self.layout.addWidget(self.plotDecibleProbabilities.view, 4, 2)
        self.nextEnergyButton = QtWidgets.QPushButton("Next")
        self.previousEnergyButton = QtWidgets.QPushButton("Previous")
        self.nextEnergyButton.clicked.connect(self.nextEnergy)
        self.previousEnergyButton.clicked.connect(self.previousEnergy)
        self.layout.addWidget(self.nextEnergyButton, 5, 2)
        self.layout.addWidget(self.previousEnergyButton, 5, 1)
        self.mainWidget.show()
        self.mainWidget.resize(1200, 800)

    def setLabels(self): 
        self.energyIndexLabel.setText(
                "Current Energy Index: " \
                + str(self.currentEnergyIndex) \
            )
        self.energyValueLabel.setText(
                "Current Energy: " \
                + "{:<15}".format(self.waves.energyValues[self.currentEnergyIndex])
            )

    def nextEnergy(self): 
        self.setLabels()
        self.plotWaves(self.currentEnergyIndex + 1 \
                if self.currentEnergyIndex < (len(self.waves.waveFunctions) - 1) \
                else self.currentEnergyIndex
            )

    def previousEnergy(self): 
        self.setLabels()
        self.plotWaves(self.currentEnergyIndex - 1 if self.currentEnergyIndex > 0 else 0)

    def plotWaves(self, currentEnergyIndex): 
        self.currentEnergyIndex = currentEnergyIndex
        self.plotWaveFunction.newVolumePlot(self.waves.waveFunctions[currentEnergyIndex])
        self.plotProbabilities.newVolumePlot(self.waves.probabilities[currentEnergyIndex])
        self.plotDecibleProbabilities.newVolumePlot(self.waves.decibleProbabilities[currentEnergyIndex])

class Plot3D: 
    def __init__(
               self, 
               currentEnergyIndex : int, 
               grid : MeshGrid, 
               potential : np.ndarray, 
               waves : WaveFunctions, 
               properties : list[bool] = [True, True, True, True], 
               pointSize = .001, 
               alpha = .6, 
               antiAliasing = False
           ):
        self.figure = None
        self.currentEnergyIndex = currentEnergyIndex
        self.grid = grid
        self.potential = potential
        self.waves = waves
        self.properties = properties 
        self.pointSize = pointSize
        self.alpha = alpha
        self.antiAliasing = antiAliasing
        self.scatterPlots = []
        self.axis = []
        self.colorBars = []
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
                if len(self.axis) <= ii: 
                    self.axis.append(self.figure.add_subplot(2, 2, ii + 1, projection="3d"))
                    self.scatterPlots.append(self.axis[-1].scatter3D(
                            self.grid.x, 
                            self.grid.y, 
                            self.grid.z, 
                            c = toPlot[ii], 
                            cmap = cm.seismic, 
                            s = self.pointSize, 
                            alpha = self.alpha, 
                            antialiased = self.antiAliasing
                        ))
                    self.axis[-1].set_title(titles[ii])
                    self.colorBars.append(self.figure.colorbar(self.scatterPlots[-1]))
                else: 
                    self.scatterPlots[ii] = self.axis[ii].scatter3D(
                            self.grid.x, 
                            self.grid.y, 
                            self.grid.z, 
                            c = toPlot[ii], 
                            cmap = cm.seismic, 
                            s = self.pointSize, 
                            alpha = self.alpha, 
                            antialiased = self.antiAliasing
                        )
                    self.colorBars[ii].remove()
                    self.colorBars[ii] = self.figure.colorbar(self.scatterPlots[ii])
        print("Done Plotting Energy")
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

