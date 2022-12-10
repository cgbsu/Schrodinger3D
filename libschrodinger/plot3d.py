import pyqtgraph as pg
import pyqtgraph.opengl as pggl
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from libschrodinger.numerov3d import DimensionIndex
from libschrodinger.numerov3d import MeshGrid

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


