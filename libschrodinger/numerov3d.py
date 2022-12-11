import types
from enum import Enum
from functools import partial
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
#from scipy import sparse
#import scipy.sparse.linalg as scipylin
from scipy.sparse.linalg import eigsh
import cupyx.scipy.sparse.linalg as cplin

from cupyx.scipy import sparse

class EigenValueTypes(Enum):
    LARGEST_MAGNITUDE = "LM"
    SMALLEST_MAGNITUDE = "SM"
    LARGEST_ALGEBRAIC = "LA"
    SMALLEST_ALGEBRAIC = "SA"
    HALF_SPECTRUM = "BE"

class DimensionIndex(Enum):
    X = 0
    Y = 1
    Z = 2
    W = 3

class WaveFunctions:
    def __init__(
                self, 
                shape : tuple[int], 
                energyValues : np.array, 
                eigenVectors : np.ndarray, 
                doNotComputeExtra : bool = False
            ):
        self.shape : tuple[int] = shape
        self.pointCount = shape[0]
        self.dimensions = len(shape)
        self.energyValues : np.array = energyValues
        self.waveFunctions : np.ndarray = np.array(list(map(
                lambda transposedWaveFunction : transposedWaveFunction.reshape(self.shape), 
                eigenVectors.T.get()
            )))
        if doNotComputeExtra == True: 
            self.probabilities = None
            self.decibleProbabilities = None
        else: 
            self.probabilities = self.waveFunctions * np.conjugate(self.waveFunctions)
            self.decibleProbabilities = 10 * np.log10(self.probabilities)

class MeshGrid: 
    def __init__(self, gridDimensionalComponents : tuple[np.ndarray], pointCount : int, length : float): 
        self.pointCount = pointCount
        self.length = length
        self.gridDimensionalComponents : tuple[np.ndarray] = gridDimensionalComponents 
        self.dimensions = len(self.gridDimensionalComponents)
        for dimension_ in list(DimensionIndex.__members__): 
            dimension = getattr(DimensionIndex, dimension_)
            if self.dimensions > dimension.value: 
                setattr(self, dimension.name.lower(), self.gridDimensionalComponents[dimension.value])
        self.asArray = None
    def toArray(self) -> np.array: 
        self.asArray = np.column_stack(np.array([
                component.ravel() \
                for component in self.gridDimensionalComponents
            ])).ravel()
        return self.asArray

def makeLinspaceGrid(pointCount : int, length : float, dimensions : int, componentType : type = float) -> MeshGrid: 
    spaces : tuple[np.array] = tuple((np.linspace(0, length, pointCount, dtype = componentType) for ii in range(dimensions)))
    return MeshGrid(np.meshgrid(*spaces), pointCount, length)

def makeMappingMatrix(pointCount : int, dimensions : int) -> np.ndarray:
    ones = np.ones([pointCount])
    baseMappingMatrix = sparse.spdiags(
        np.array([ones, -2 * ones, ones]), 
        np.array([-1, 0, 1]), 
        pointCount, 
        pointCount
    )
    mappingMatrix = baseMappingMatrix
    for ii in range(1, dimensions): 
        mappingMatrix = sparse.kronsum(mappingMatrix, baseMappingMatrix)
    return mappingMatrix, baseMappingMatrix

def kineticEnergyOperator(mappingMatrix : np.ndarray) -> np.ndarray: 
    return (-1.0 / 2.0) * mappingMatrix

def potentialEnergyOperator(potential : np.ndarray, pointCount : int, dimensions : int) -> np.ndarray: 
    return sparse.diags(potential.reshape(pointCount ** dimensions), (0))

def makeHamiltonian(
            potential : np.ndarray, 
            pointCount : int, 
            dimensions : int, 
            mappingMatrix : np.ndarray
        ) -> np.ndarray: 
    return kineticEnergyOperator(mappingMatrix) + potentialEnergyOperator(
            potential, 
            pointCount, 
            dimensions
        )

def computeWaveFunction(
            potential : np.ndarray, 
            energyCount : int = 10, 
            eigenValueType : EigenValueTypes = EigenValueTypes.LARGEST_MAGNITUDE, # EigenValueTypes.SMALLEST_MAGNITUDE
            #center = 0, 
            #gpuAccelerated = True
        ) -> WaveFunctions: 
    dimensions : int = len(potential.shape)
    pointCount : int = potential.shape[0]
    for cardinality in potential.shape: 
        assert cardinality == pointCount, "All dimensions of potential need to have the same number of elements"
    mappingMatrix, _ = makeMappingMatrix(pointCount, dimensions)
    hamiltonian = makeHamiltonian(potential, pointCount, dimensions, mappingMatrix)
    
    #if gpuAccelerated == False: 
    eigenStates = cplin.eigsh( 
            hamiltonian, 
            k = energyCount, 
            which = eigenValueType.value, 
            #sigma = sigma
        )
    #else: 
    #    eigenStates = cupylin.eigsh(
    #            hamiltonian, 
    #            k = energyCount, 
    #            which = eigenValueType.value, 
    #        )
    return WaveFunctions(potential.shape, *eigenStates)

