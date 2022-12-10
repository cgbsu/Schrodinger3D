import pyqtgraph as pg
import pyqtgraph.opengl as pggl
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from libschrodinger.numerov3d import DimensionIndex
from libschrodinger.numerov3d import MeshGrid

class Plot3D: 
    def __init__(self, application, data : np.ndarray):
        self.application = application
        #self.grid = grid.toArray()
        self.data = data
        self.dataMax = np.max(np.abs(data)) # USE Min?
        self.dataMax = self.dataMax if abs(self.dataMax) > 0 else 1
        self.colors = np.empty(data.shape + (4,), dtype=np.ubyte)
        self.colors[..., 0] = -10 * np.log10(self.data)
        self.colors[..., 3] = 150
        self.view = pggl.GLViewWidget()
        self.view.show()
        self.grid = pggl.GLGridItem()
        self.plot = pggl.GLVolumeItem(self.colors)
        self.axis = pggl.GLAxisItem()
        self.view.addItem(self.grid)
        self.view.addItem(self.plot)
        self.view.addItem(self.axis)


