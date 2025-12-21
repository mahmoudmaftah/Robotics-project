"""
Grid-based abstraction for symbolic control.

Discretizes the continuous state space into a grid of cells,
enabling graph-based planning algorithms for reach-avoid specifications.
"""

import numpy as np
import networkx as nx
from typing import Tuple, List, Optional, Set, Dict, Any
from shapely.geometry import Polygon, Point, box


class GridAbstraction:
    """
    Grid-based state space abstraction.

    Discretizes a 2D state space into rectangular cells and
    constructs a transition graph based on robot dynamics.

    Attributes:
        bounds: State space bounds [[x_min, x_max], [y_min, y_max]]
        resolution: Grid resolution [nx, ny]
        cells: Cell centers
        graph: Transition graph (networkx DiGraph)
    """

    def __init__(self, bounds: np.ndarray, resolution: Tuple[int, int],
                 model=None):
        """
        Initialize grid abstraction.

        Args:
            bounds: State bounds (2x2 array: [[x_min, x_max], [y_min, y_max]])
            resolution: Number of cells in each dimension (nx, ny)
            model: Robot model for transition computation
        """
        self.bounds = np.asarray(bounds)
        self.resolution = resolution
        self.model = model

        # Compute cell dimensions
        self.cell_width = (bounds[0, 1] - bounds[0, 0]) / resolution[0]
        self.cell_height = (bounds[1, 1] - bounds[1, 0]) / resolution[1]

        # Create cell centers
        x_centers = np.linspace(bounds[0, 0] + self.cell_width/2,
                                bounds[0, 1] - self.cell_width/2,
                                resolution[0])
        y_centers = np.linspace(bounds[1, 0] + self.cell_height/2,
                                bounds[1, 1] - self.cell_height/2,
                                resolution[1])

        self.x_centers = x_centers
        self.y_centers = y_centers

        # Create cell center array
        xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
        self.cells = np.stack([xx.flatten(), yy.flatten()], axis=1)

        # Total number of cells
        self.n_cells = resolution[0] * resolution[1]

        # Create transition graph
        self.graph = nx.DiGraph()
        self._build_graph()

        # Labeling
        self._goal_cells: Set[int] = set()
        self._obstacle_cells: Set[int] = set()
        self._safe_cells: Set[int] = set(range(self.n_cells))

    def _build_graph(self) -> None:
        """Build transition graph with 8-connectivity."""
        nx_cells, ny_cells = self.resolution

        for i in range(self.n_cells):
            self.graph.add_node(i, center=self.cells[i])

        # Add edges (8-connected grid)
        for ix in range(nx_cells):
            for iy in range(ny_cells):
                cell_id = self._coords_to_id(ix, iy)

                # 8 neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue

                        nix, niy = ix + dx, iy + dy
                        if 0 <= nix < nx_cells and 0 <= niy < ny_cells:
                            neighbor_id = self._coords_to_id(nix, niy)
                            # Edge weight based on distance
                            if abs(dx) + abs(dy) == 1:
                                weight = 1.0
                            else:
                                weight = np.sqrt(2)
                            self.graph.add_edge(cell_id, neighbor_id, weight=weight)

    def _coords_to_id(self, ix: int, iy: int) -> int:
        """Convert grid coordinates to cell ID."""
        return ix * self.resolution[1] + iy

    def _id_to_coords(self, cell_id: int) -> Tuple[int, int]:
        """Convert cell ID to grid coordinates."""
        ix = cell_id // self.resolution[1]
        iy = cell_id % self.resolution[1]
        return ix, iy

    def state_to_cell(self, state: np.ndarray) -> int:
        """
        Map continuous state to discrete cell ID.

        Args:
            state: Continuous state [x, y]

        Returns:
            Cell ID (or -1 if outside bounds)
        """
        x, y = state[0], state[1]

        if x < self.bounds[0, 0] or x > self.bounds[0, 1]:
            return -1
        if y < self.bounds[1, 0] or y > self.bounds[1, 1]:
            return -1

        ix = int((x - self.bounds[0, 0]) / self.cell_width)
        iy = int((y - self.bounds[1, 0]) / self.cell_height)

        # Clamp to valid range
        ix = min(ix, self.resolution[0] - 1)
        iy = min(iy, self.resolution[1] - 1)

        return self._coords_to_id(ix, iy)

    def get_cell_center(self, cell_id: int) -> np.ndarray:
        """
        Get cell center coordinates.

        Args:
            cell_id: Cell ID

        Returns:
            Cell center coordinates
        """
        return self.cells[cell_id].copy()

    def cell_to_state(self, cell_id: int) -> np.ndarray:
        """
        Get cell center as continuous state.

        Args:
            cell_id: Cell ID

        Returns:
            Cell center coordinates
        """
        return self.cells[cell_id].copy()

    def get_cell_polygon(self, cell_id: int) -> Polygon:
        """
        Get cell as Shapely polygon.

        Args:
            cell_id: Cell ID

        Returns:
            Shapely Polygon
        """
        center = self.cells[cell_id]
        half_w = self.cell_width / 2
        half_h = self.cell_height / 2

        return box(center[0] - half_w, center[1] - half_h,
                   center[0] + half_w, center[1] + half_h)

    def set_goal_region(self, polygon: np.ndarray) -> Set[int]:
        """
        Mark cells as goal region.

        Args:
            polygon: Goal region vertices (Nx2 array)

        Returns:
            Set of goal cell IDs
        """
        goal_poly = Polygon(polygon)
        self._goal_cells = set()

        for cell_id in range(self.n_cells):
            cell_poly = self.get_cell_polygon(cell_id)
            if goal_poly.intersects(cell_poly):
                self._goal_cells.add(cell_id)

        return self._goal_cells

    # define a function that adds one obstacle region

    def add_obstacle(self, polygon: np.ndarray) -> Set[int]:
        """
        Add an obstacle region by marking cells.

        Args:
            polygon: Obstacle region vertices (Nx2 array)

        Returns:
            Set of newly added obstacle cell IDs
        """
        obs_poly = Polygon(polygon)
        new_obstacles = set()

        for cell_id in range(self.n_cells):
            cell_poly = self.get_cell_polygon(cell_id)
            if obs_poly.intersects(cell_poly):
                new_obstacles.add(cell_id)

        self._obstacle_cells.update(new_obstacles)
        # Update safe cells
        self._safe_cells = set(range(self.n_cells)) - self._obstacle_cells

        return new_obstacles

    def set_obstacles(self, polygons: List[np.ndarray]) -> Set[int]:
        """
        Mark cells as obstacles.

        Args:
            polygons: List of obstacle vertex arrays

        Returns:
            Set of obstacle cell IDs
        """
        self._obstacle_cells = set()

        for obs_vertices in polygons:
            obs_poly = Polygon(obs_vertices)
            for cell_id in range(self.n_cells):
                cell_poly = self.get_cell_polygon(cell_id)
                if obs_poly.intersects(cell_poly):
                    self._obstacle_cells.add(cell_id)

        # Update safe cells
        self._safe_cells = set(range(self.n_cells)) - self._obstacle_cells

        return self._obstacle_cells

    @property
    def goal_cells(self) -> Set[int]:
        """Get goal cell IDs."""
        return self._goal_cells

    @property
    def obstacle_cells(self) -> Set[int]:
        """Get obstacle cell IDs."""
        return self._obstacle_cells

    @property
    def safe_cells(self) -> Set[int]:
        """Get safe (non-obstacle) cell IDs."""
        return self._safe_cells

    def get_safe_graph(self) -> nx.DiGraph:
        """
        Get subgraph with only safe cells.

        Returns:
            Graph with obstacle cells removed
        """
        return self.graph.subgraph(self._safe_cells).copy()

    def visualize(self, ax=None, show_labels: bool = False):
        """
        Visualize grid abstraction.

        Args:
            ax: Matplotlib axes
            show_labels: Show cell IDs

        Returns:
            Matplotlib axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Draw cells
        for cell_id in range(self.n_cells):
            center = self.cells[cell_id]
            rect = Rectangle((center[0] - self.cell_width/2,
                              center[1] - self.cell_height/2),
                             self.cell_width, self.cell_height,
                             fill=False, edgecolor='lightgray')

            if cell_id in self._obstacle_cells:
                rect.set_facecolor('gray')
                rect.set_fill(True)
                rect.set_alpha(0.5)
            elif cell_id in self._goal_cells:
                rect.set_facecolor('green')
                rect.set_fill(True)
                rect.set_alpha(0.3)

            ax.add_patch(rect)

            if show_labels:
                ax.text(center[0], center[1], str(cell_id),
                       ha='center', va='center', fontsize=6)

        ax.set_xlim(self.bounds[0])
        ax.set_ylim(self.bounds[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        return ax
