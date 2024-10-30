# This work is under the Copyright Christopher A. Greeley (2024) and it is distributed
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

def gaussian(
            positions : np.ndarray, 
            offset : float, 
            variance : float
        ) -> np.ndarray: 
    exponent = (-1.0 / 2.0) * ((positions - offset) ** 2) / (2.0 * variance ** 2)
    scalar = 1.0 / (np.sqrt(2.0 * np.pi) * variance)
    return scalar * np.exp(exponent)
    

def gaussian2d(
            grid, 
            xOffset : float, 
            yOffset : float, 
            variance : float
        ) -> np.ndarray: 
    return gaussian(grid.x, xOffset, variance) * gaussian(grid.x, yOffset, variance)

def hydrogenAtom(
            grid, 
            centerX : float, 
            centerY : float, 
            bottom : float
        ) -> np.ndarray: 
    distance = 1 / (np.sqrt(((xPosition - centerX) ** 2) + ((yPosition - centerY) ** 2)) + bottom)
    return distance

def tunnelingCase(
            grid, 
            barrierPosition : float, 
            barrierWidth : float, 
            potentialHeight : float
        ) -> np.ndarray: 
    potentials = np.zeros(xPositions.shape)
    return np.where(
            (grid.x <= (barrierPosition + barrierWidth)) \
                    & (grid.x >= (barrierPosition - barrierWidth)), 
            potentialHeight, 
            potentials
        )

def finiteSquareWell(
            grid, 
            potentialHeight : float, 
            width : float, 
            height : float
        ) -> np.ndarray: 
    potential : np.ndarray = np.zeros(xPositions.shape)
    potential = np.where((grid.x <= width) | (grid.x >= (1 - width)), potentialHeight, potential)
    potential = np.where((grid.y <= height) | (grid.y >= (1 - height)), potentialHeight, potential)
    return potential

def finiteCircularWell(
            grid, 
            potentialHeight : float, 
            radius : float, 
            centerX : float = .5, 
            centerY : float = .5
        ) -> np.ndarray: 
    distances : float = np.sqrt(((grid.x - centerX) ** 2) + ((grid.y - centerY) ** 2))
    potential : np.ndarray = np.zeros(grid.x.shape)
    potential = np.where(distances >= radius, potentialHeight, potential)
    return potential

def stairwell(
            grid, 
            unitPotentialHeight : float, 
            widthRatios : list[float], 
            heightRatios : list[float], 
            unitLength : float  = 1
        ) -> np.ndarray: 
    assert len(heightRatios) == len(widthRatios)
    potential : np.ndarray = np.zeros(grid.x.shape)
    previousLength : float = 0
    lengthRatio : float = 0
    for ii in range(len(widthRatios)): 
        lengthRatio += widthRatios[ii]
        length = lengthRatio * unitLength
        potential = np.where(
                (grid.x <= (length))
                        & (grid.x >= previousLength), 
                heightRatios[ii] * unitPotentialHeight, 
                potential
            )
        previousLength = length
    return potential

