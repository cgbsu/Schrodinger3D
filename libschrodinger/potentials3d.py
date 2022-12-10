import numpy as np

def hydrogenAtom(
            grid, 
            centerX : float, 
            centerY : float, 
            centerZ : float, 
            bottom : float, 
            potential : float
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
            centerZ : float, 
            width : float, 
            height : float, 
            potential : float, 
        ) -> np.ndarray: 
    potential_ = np.zeros(grid.x.shape)
    return np.where(
            ~((np.abs(centerX - grid.x) < width) \
                    & (np.abs(centerZ - grid.z) < height)), 
            potential, 
            potential_
        )
