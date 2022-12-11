import numpy as np
from dataclasses import dataclass

@dataclass
class Stairwell: 
    heights : list[float] 
    widths : list[float] 
    unitPotential : float
    unitWidth : float

UNIFORM_STAIRWELL = Stairwell(
        [1.0 / 3.0, 2.0 / 3.0, 1.0], 
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 
        1.0, 
        1.0
    )


def hydrogenAtom(
            grid, 
            centerX : float = .5, 
            centerY : float = .5, 
            centerZ : float = .5, 
            bottom : float = 1, 
            potential : float = 1
        ) -> np.ndarray: 
    return potential / np.sqrt(
            (grid.x - centerX) ** 2 \
            + (grid.y - centerY) ** 2 \
            + (grid.z - centerZ) ** 2 \
            + bottom ** 2 \
        )

def tunnelingCase(
            grid, 
            centerX : float, 
            width : float, 
            potential : float, 
            length : float = 1.0
        ) -> np.ndarray: 
    centerX *= length
    width *= length
    return np.where(
            (grid.x <= (centerX + width)) & (grid.x >= centerX), 
            0, 
            potential, 
        )

def constantPotentialRegions(
            grid, 
            widths : float, 
            heights : float,  
            unitPotential : float = 1.0, 
            unitWidth : float = 1.0
        ) -> np.ndarray: 
    potential = np.zeros(grid.x.shape)
    length = 0
    for ii in range(len(heights)): 
        width = widths[ii] * unitWidth
        height = heights[ii] * unitPotential
        print(height)
        potential = np.where(
                (grid.x >= length) & (grid.x <= (length + width)), 
                height, 
                potential
            )
        length += width
    return potential

def stairwell(grid, configuration : Stairwell = UNIFORM_STAIRWELL) -> np.ndarray: 
    return constantPotentialRegions(
            grid, 
            configuration.widths, 
            configuration.heights, 
            configuration.unitPotential, 
            configuration.unitWidth
        )

