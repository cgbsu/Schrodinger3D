import pyqtgraph as pg
import pyqtgraph.opengl as pggl
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from libschrodinger.numerov3d import DimensionIndex
from libschrodinger.numerov3d import MeshGrid

class Plot3D: 
    MAXIMUM_32_BITS = 2 ** 32
    BASE_10_DIGITS_IN_32_BITS = np.uint32(np.ceil(np.log10(MAXIMUM_32_BITS)))
    BASE_2_DIGITS_IN_32_BITS = np.uint32(np.ceil(np.log2(MAXIMUM_32_BITS)))
    BASE_10_MAXIMUM_BASE_32_BITS = 10 ** BASE_10_DIGITS_IN_32_BITS 
    def __init__(self, application, data : np.ndarray):
        self.application = application
        #self.grid = grid.toArray()
        self.data = data
        self.dataMax = np.max(data)
        self.dataMax = self.dataMax if abs(self.dataMax) > 0 else 1
        # Graph takes grid in the format of an array, of Nx4 unsigned bytes
        # First the values must be normalized (in this case, put on a scale [-1, 1], 
        # then we must preserve as much information as possible when converting to 
        # 32 bit unsigned integers (4x8 bits)
        self.normalizedColors = np.uint32(
                np.ceil((self.data / self.dataMax)).ravel() \
                * Plot3D.BASE_10_MAXIMUM_BASE_32_BITS  # The array is RGBA values each component with 8 bits
            )
        self.view = pggl.GLViewWidget()
        #self.grids = [pggl.GLGridItem() for ii in range(3)]
        self.view.show()
        #for grid in self.grids:
        #    self.view.addItem(grid)
        self.view.addItem(pggl.GLGridItem())
        #self.grids[DimensionIndex.X.value].rotate(90, 0, 1, 0)
        #self.grids[DimensionIndex.Y.value].rotate(90, 1, 0, 0)
        self.plot = pggl.GLVolumeItem(self.normalizedColors)
        self.view.addItem(self.plot)
        self.view.addItem(pggl.GLAxisItem())

