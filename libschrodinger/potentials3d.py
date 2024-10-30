# This work is under the Copywrite Christopher A. Greeley (2024) and it is distributed
# under the No Kill Do No Harm License, a legally non-binding sumemry is as follows: 
# 
# # No Kill Do No Harm Licence – Summary
# 
# Based on version 0.3, July 2022 of the Do No Harm License
# 
# https://github.com/raisely/NoHarm
# 
# LEGALLY NON-BINDING SUMMARY OF THE TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# 
# ## Licence Grants
# 
# You're allowed
# 
# - to distribute the licensed work,
# - to create, publish, sublicense and patent derivative works and
# - to put your modifications or your derivative work under a seperate licence,
# 
# free of charge. Though, filing patent litigation leads to the loss of the patent licence. Also, the licence grants don't include the right to use the licensor's trademarks.
# 
# ## Unethical Behaviour
# 
# You may not use the licensed work if you engage in:
# 
# - human rights violations,
# - environmental destruction,
# - warfare,
# - addictive/destructive products or services or
# - actions that frustrate:
#   * peace,
#   * access to human rights,
#   * peaceful assembly and association,
#   * a sustainable environment or
#   * democratic processes
#   * abortion
#   * euthanasia
#   * human embryonic stem cell research (if human organisms are killed in the process)
# - except for actions that may be contrary to "human rights" (or interpretations thereof), do not kill and that frustrate 
#   * abortion
#   * euthanasia
#   * killing
# and; the software must never be used to kill, including: abortion, euthanasia, human stem cell research, in war, or law enforcement or as a part of any lethal weapon
# 
# ## Contributions
# 
# Contributions to the licensed work must be licensed under the exact same licence.
# 
# ## Licence Notice
# 
# When distributing the licensed work or your derivative work, you must
# 
# - include a copy of this licence,
# - retain attribution notices,
# - state changes that you made and
# - not use the names of the author and the contributors to promote your derivative work.
# 
# If the licensed work includes a "NOTICE" text file with attribution notices, you must copy those notices to:
# 
# - a "NOTICE" file within your derivative work,
# - a place within the source code or the documentation or
# - a place within a display generated by your derivative work.
# 
# ## No Warranty or Liability
# 
# The licensed work is offered on an as-is basis without any warranty or liability. You may choose to offer warranty or liability for your derivative work, but only fully on your own responsibility.
#


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
        .1, 
        1.0
    )


def hydrogenAtom(
            grid, 
            centerX : float = .5, 
            centerY : float = .5, 
            centerZ : float = .5, 
            bottom : float = 1, 
            potential : float = .05
        ) -> np.ndarray: 
    return -potential / np.sqrt(
            grid.x ** 2 \
            + grid.y ** 2 \
            + grid.z ** 2 \
            + bottom ** 2 \
        )
    #return -potential / np.sqrt(
    #        (grid.x - centerX) ** 2 \
    #        + (grid.y - ceterY) ** 2 \
    #        + (grid.z - centerZ) ** 2 \
    #        + bottom ** 2 \
    #    )

def tunnelingCase(
            grid, 
            centerX : float, 
            width : float, 
            potential : float = .1, 
            length : float = 1.0
        ) -> np.ndarray: 
    centerX *= length
    width *= length
    V = np.where(
            (grid.x <= (centerX + width)) & (grid.x >= centerX), 
            potential, 
            0
        )
    return V

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

def torus(
            grid, 
            innerRadiusRatio : float = .1, 
            outerRadiusRatio : float = .4, 
            potential : float = 1
        ) -> np.ndarray:
    length = np.abs(grid.x.min()) + np.abs(grid.x.max())
    planeTerm = np.sqrt(((grid.x / length) ** 2) + ((grid.y / length) ** 2))
    return potential * ((planeTerm - outerRadiusRatio) ** 2) + ((grid.z / length) ** 2) - (innerRadiusRatio ** 2)

