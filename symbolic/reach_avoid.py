"""
Reach-Avoid planning and control.

Implements graph-based planning for reachability specifications
while avoiding obstacles.

The reach-avoid problem:
    Given: Initial state x0, goal region G, obstacle set O
    Find: Controller that drives the system to G while avoiding O

This is formalized as finding a path in the abstraction graph
from the initial cell to a goal cell, avoiding obstacle cells.
"""

import numpy as np
import networkx as nx
from typing import Optional, List, Tuple, Dict, Any, Set
from .grid_abstraction import GridAbstraction
from controllers.base import Controller


class ReachAvoidPlanner:
    """
    Graph-based reach-avoid planner.

    Uses Dijkstra's algorithm on the grid abstraction to find
    shortest paths to goal while avoiding obstacles.

    The planner computes:
    1. Value function V(cell) = minimum cost to reach goal from cell
    2. Policy π(cell) = optimal next cell to visit
    """

    def __init__(self, abstraction: GridAbstraction):
        """
        Initialize planner.

        Args:
            abstraction: Grid abstraction of state space
        """
        self.abstraction = abstraction
        self._value_function: Dict[int, float] = {}
        self._policy: Dict[int, int] = {}
        self._path_cache: Dict[int, List[int]] = {}

    def compute_value_function(self) -> Dict[int, float]:
        """
        Compute value function using backward reachability.

        V(cell) = min cost to reach any goal cell from cell
        V(goal) = 0, V(obstacle) = inf

        Returns:
            Dictionary mapping cell IDs to values
        """
        safe_graph = self.abstraction.get_safe_graph()
        goal_cells = self.abstraction.goal_cells

        if not goal_cells:
            raise ValueError("No goal cells defined")

        # Initialize value function
        self._value_function = {cell: np.inf for cell in range(self.abstraction.n_cells)}

        # Goal cells have zero cost
        for goal in goal_cells:
            self._value_function[goal] = 0.0

        # Create reverse graph for backward computation
        reverse_graph = safe_graph.reverse()

        # Multi-source Dijkstra from goal cells
        for goal in goal_cells:
            if goal in safe_graph:
                try:
                    distances = nx.single_source_dijkstra_path_length(
                        reverse_graph, goal, weight='weight')
                    for cell, dist in distances.items():
                        self._value_function[cell] = min(
                            self._value_function[cell], dist)
                except nx.NetworkXError:
                    pass

        return self._value_function

    def compute_policy(self) -> Dict[int, int]:
        """
        Compute optimal policy from value function.

        π(cell) = argmin_{neighbor} V(neighbor) + cost(cell, neighbor)

        Returns:
            Dictionary mapping cell IDs to next cell IDs
        """
        if not self._value_function:
            self.compute_value_function()

        self._policy = {}
        safe_graph = self.abstraction.get_safe_graph()

        for cell in self.abstraction.safe_cells:
            if cell in self.abstraction.goal_cells:
                self._policy[cell] = cell  # Stay at goal
                continue

            if cell not in safe_graph:
                continue

            # Find best neighbor
            best_next = None
            best_value = np.inf

            for neighbor in safe_graph.successors(cell):
                edge_cost = safe_graph[cell][neighbor].get('weight', 1.0)
                total_cost = edge_cost + self._value_function.get(neighbor, np.inf)
                if total_cost < best_value:
                    best_value = total_cost
                    best_next = neighbor

            if best_next is not None:
                self._policy[cell] = best_next

        return self._policy

    def get_path(self, start_cell: int) -> Optional[List[int]]:
        """
        Get optimal path from start cell to goal.

        Args:
            start_cell: Starting cell ID

        Returns:
            List of cell IDs forming path, or None if unreachable
        """
        if start_cell in self._path_cache:
            return self._path_cache[start_cell]

        if not self._policy:
            self.compute_policy()

        if start_cell in self.abstraction.obstacle_cells:
            return None

        if self._value_function.get(start_cell, np.inf) == np.inf:
            return None

        path = [start_cell]
        current = start_cell
        visited = {start_cell}

        while current not in self.abstraction.goal_cells:
            if current not in self._policy:
                return None

            next_cell = self._policy[current]

            if next_cell in visited:
                # Cycle detected
                return None

            path.append(next_cell)
            visited.add(next_cell)
            current = next_cell

            if len(path) > self.abstraction.n_cells:
                # Safety check
                return None

        self._path_cache[start_cell] = path
        return path

    def get_path_from_state(self, state: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Get path as continuous waypoints.

        Args:
            state: Starting continuous state

        Returns:
            List of waypoint states
        """
        cell = self.abstraction.state_to_cell(state)
        if cell < 0:
            return None

        cell_path = self.get_path(cell)
        if cell_path is None:
            return None

        # Convert to continuous states
        return [self.abstraction.cell_to_state(c) for c in cell_path]


class ReachAvoidController(Controller):
    """
    Controller that follows a reach-avoid plan.

    Combines symbolic planning with low-level proportional control
    to track waypoints from the plan.

    The controller:
    1. Uses the planner to get the optimal path
    2. Tracks waypoints using proportional control
    3. Switches to next waypoint when close enough
    """

    def __init__(self, planner: ReachAvoidPlanner,
                 waypoint_tolerance: float = 0.5,
                 kp: float = 0.8,
                 name: str = "ReachAvoid"):
        """
        Initialize reach-avoid controller.

        Args:
            planner: ReachAvoidPlanner instance
            waypoint_tolerance: Distance to switch waypoints
            kp: Proportional gain for waypoint tracking
            name: Controller name
        """
        super().__init__(name)
        self.planner = planner
        self.waypoint_tolerance = waypoint_tolerance
        self.kp = kp

        # Internal state
        self._waypoints: Optional[List[np.ndarray]] = None
        self._current_waypoint_idx: int = 0
        self._path_computed: bool = False

    def reset(self) -> None:
        """Reset controller state."""
        self._waypoints = None
        self._current_waypoint_idx = 0
        self._path_computed = False

    def _compute_path(self, x: np.ndarray) -> bool:
        """
        Compute path from current state.

        Args:
            x: Current state

        Returns:
            True if path found
        """
        waypoints = self.planner.get_path_from_state(x)
        if waypoints is None:
            return False

        self._waypoints = waypoints
        self._current_waypoint_idx = 0
        self._path_computed = True
        return True

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute control to follow reach-avoid plan.

        Args:
            x: Current state
            t: Current time

        Returns:
            u: Control input
            diagnostics: Path and waypoint info
        """
        x = np.asarray(x, dtype=float)

        # Compute path on first call
        if not self._path_computed:
            if not self._compute_path(x):
                # No path found - return zero control
                return np.zeros(2), {
                    'error': x.copy(),
                    'error_norm': np.linalg.norm(x),
                    'lyapunov': np.linalg.norm(x)**2,
                    'status': 'no_path',
                    'waypoint': None,
                    'waypoint_idx': -1
                }

        # Get current waypoint
        if self._waypoints is None or len(self._waypoints) == 0:
            return np.zeros(2), {'status': 'no_waypoints'}

        waypoint = self._waypoints[min(self._current_waypoint_idx,
                                        len(self._waypoints) - 1)]

        # Check if we reached the waypoint
        dist_to_waypoint = np.linalg.norm(x[:2] - waypoint[:2])

        if dist_to_waypoint < self.waypoint_tolerance:
            if self._current_waypoint_idx < len(self._waypoints) - 1:
                self._current_waypoint_idx += 1
                waypoint = self._waypoints[self._current_waypoint_idx]

        # Proportional control to waypoint
        error = x[:2] - waypoint[:2]
        u = -self.kp * error

        # Lyapunov function: distance to final goal
        final_waypoint = self._waypoints[-1]
        dist_to_goal = np.linalg.norm(x[:2] - final_waypoint[:2])
        V = dist_to_goal**2

        diagnostics = {
            'error': error.copy(),
            'error_norm': np.linalg.norm(error),
            'lyapunov': V,
            'dist_to_waypoint': dist_to_waypoint,
            'dist_to_goal': dist_to_goal,
            'waypoint': waypoint.copy(),
            'waypoint_idx': self._current_waypoint_idx,
            'total_waypoints': len(self._waypoints),
            'status': 'tracking'
        }

        return u, diagnostics

    def get_lyapunov_function(self, x: np.ndarray) -> Optional[float]:
        """
        Compute Lyapunov function (distance to goal).

        Args:
            x: State vector

        Returns:
            V(x) = ||x - goal||^2
        """
        if self._waypoints is None or len(self._waypoints) == 0:
            return None
        final_waypoint = self._waypoints[-1]
        return float(np.linalg.norm(x[:2] - final_waypoint[:2])**2)

    def visualize_plan(self, ax=None):
        """
        Visualize the planned path.

        Args:
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Draw grid
        self.planner.abstraction.visualize(ax)

        # Draw waypoints
        if self._waypoints is not None:
            waypoints = np.array(self._waypoints)
            ax.plot(waypoints[:, 0], waypoints[:, 1], 'b-o',
                   linewidth=2, markersize=8, label='Plan')

        return ax
