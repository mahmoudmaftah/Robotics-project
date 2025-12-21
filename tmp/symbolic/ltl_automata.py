"""
LTL (Linear Temporal Logic) and Buchi Automata for Symbolic Control.

Implements:
1. Buchi automata for LTL specifications
2. Product automaton (system x specification)
3. Accepting cycle detection for recurrence/patrolling
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Callable
from collections import defaultdict
import heapq


class BuchiAutomaton:
    """
    Buchi automaton for representing LTL specifications.

    A Buchi automaton accepts infinite words where accepting states
    are visited infinitely often.

    Components:
        - States Q
        - Initial states Q0 subset of Q
        - Transitions delta: Q x Sigma -> 2^Q
        - Accepting states F subset of Q
    """

    def __init__(self, name: str = "Buchi"):
        """
        Initialize empty Buchi automaton.

        Args:
            name: Name/description of the specification
        """
        self.name = name
        self.states: Set[str] = set()
        self.initial_states: Set[str] = set()
        self.accepting_states: Set[str] = set()
        # Transitions: state -> list of (guard_function, next_state)
        self.transitions: Dict[str, List[Tuple[Callable, str]]] = defaultdict(list)
        self.alphabet: Set[str] = set()  # Atomic propositions

    def add_state(self, state: str, initial: bool = False, accepting: bool = False):
        """Add a state to the automaton."""
        self.states.add(state)
        if initial:
            self.initial_states.add(state)
        if accepting:
            self.accepting_states.add(state)

    def add_transition(self, from_state: str, guard: Callable, to_state: str):
        """
        Add a transition.

        Args:
            from_state: Source state
            guard: Function that takes atomic propositions dict and returns bool
            to_state: Target state
        """
        self.transitions[from_state].append((guard, to_state))

    def get_successors(self, state: str, props: Dict[str, bool]) -> Set[str]:
        """
        Get successor states given current state and propositions.

        Args:
            state: Current automaton state
            props: Dictionary of atomic proposition evaluations

        Returns:
            Set of successor states
        """
        successors = set()
        for guard, next_state in self.transitions[state]:
            if guard(props):
                successors.add(next_state)
        return successors

    def __repr__(self):
        return (f"BuchiAutomaton('{self.name}', states={len(self.states)}, "
                f"accepting={len(self.accepting_states)})")


class PatrolSpecification(BuchiAutomaton):
    """
    Buchi automaton for patrolling/recurrence specification.

    Specification: GF(region1) & GF(region2) & ... & GF(regionN)
    (Globally, eventually visit each region infinitely often)
    """

    def __init__(self, region_names: List[str]):
        """
        Create patrolling specification.

        Args:
            region_names: Names of regions to patrol
        """
        super().__init__(f"Patrol({', '.join(region_names)})")
        self.region_names = region_names
        self._build_automaton()

    def _build_automaton(self):
        """Build Buchi automaton for patrolling specification."""
        n = len(self.region_names)

        # States: track which regions still need to be visited in current cycle
        # State encoding: binary string indicating remaining regions
        for i in range(2**n):
            state_name = f"q{i}"
            self.add_state(state_name,
                          initial=(i == 2**n - 1),  # All regions to visit initially
                          accepting=(i == 0))       # All visited = accepting

        # Transitions
        for i in range(2**n):
            from_state = f"q{i}"

            # For each possible proposition evaluation
            for region_idx, region_name in enumerate(self.region_names):
                # If we're in a region, mark it as visited
                bit = 1 << region_idx

                def make_guard(ridx, rname):
                    def guard(props):
                        return props.get(rname, False)
                    return guard

                # Transition when visiting this region
                guard = make_guard(region_idx, region_name)

                # Clear the bit for this region
                new_state_idx = i & ~bit

                # If all regions visited (state 0), reset to full
                if new_state_idx == 0 and i != 0:
                    to_state = f"q{2**n - 1}"  # Reset cycle
                else:
                    to_state = f"q{new_state_idx}"

                self.transitions[from_state].append((guard, to_state))

        # Self-loops when not in any target region
        for i in range(2**n):
            from_state = f"q{i}"

            def not_in_any(props):
                return not any(props.get(r, False) for r in self.region_names)

            self.transitions[from_state].append((not_in_any, from_state))


class SequenceSpecification(BuchiAutomaton):
    """
    Buchi automaton for visiting regions in sequence.

    Specification: F(r1 & F(r2 & F(r3 & ...)))
    (Eventually visit regions in order, then repeat)
    """

    def __init__(self, region_names: List[str]):
        """
        Create sequence specification.

        Args:
            region_names: Names of regions in visitation order
        """
        super().__init__(f"Sequence({' -> '.join(region_names)})")
        self.region_names = region_names
        self._build_automaton()

    def _build_automaton(self):
        """Build Buchi automaton for sequence specification."""
        n = len(self.region_names)

        # States: current position in sequence
        for i in range(n):
            self.add_state(f"q{i}",
                          initial=(i == 0),
                          accepting=(i == 0))  # Accept when completing cycle

        # Transitions
        for i in range(n):
            from_state = f"q{i}"
            target_region = self.region_names[i]
            next_idx = (i + 1) % n

            def make_guard(region):
                def guard(props):
                    return props.get(region, False)
                return guard

            def make_not_guard(region):
                def guard(props):
                    return not props.get(region, False)
                return guard

            # Progress when in target region
            self.transitions[from_state].append(
                (make_guard(target_region), f"q{next_idx}")
            )

            # Stay when not in target region
            self.transitions[from_state].append(
                (make_not_guard(target_region), from_state)
            )


class ProductAutomaton:
    """
    Product of system abstraction and Buchi automaton.

    Used for finding accepting runs that satisfy LTL specifications.
    """

    def __init__(self, abstraction, buchi: BuchiAutomaton,
                 labeling: Callable[[int], Dict[str, bool]]):
        """
        Create product automaton.

        Args:
            abstraction: Grid abstraction with transition system
            buchi: Buchi automaton for specification
            labeling: Function mapping cell indices to atomic propositions
        """
        self.abstraction = abstraction
        self.buchi = buchi
        self.labeling = labeling

        # Product states: (cell_index, buchi_state)
        self.product_states: Set[Tuple[int, str]] = set()
        self.transitions: Dict[Tuple[int, str], List[Tuple[int, str]]] = defaultdict(list)

        self._build_product()

    def _build_product(self):
        """Build the product automaton."""
        # Build product states and transitions
        for cell_idx in self.abstraction.safe_cells:
            props = self.labeling(cell_idx)

            for buchi_state in self.buchi.states:
                product_state = (cell_idx, buchi_state)
                self.product_states.add(product_state)

                # Get Buchi successors
                buchi_succs = self.buchi.get_successors(buchi_state, props)

                # Get system successors
                if hasattr(self.abstraction, 'get_successors'):
                    sys_succs = self.abstraction.get_successors(cell_idx)
                else:
                    sys_succs = self._get_cell_successors(cell_idx)

                # Product transitions
                for sys_succ in sys_succs:
                    succ_props = self.labeling(sys_succ)
                    for buchi_succ in buchi_succs:
                        new_buchi_succs = self.buchi.get_successors(buchi_succ, succ_props)
                        for new_b in new_buchi_succs:
                            self.transitions[product_state].append((sys_succ, new_b))

    def _get_cell_successors(self, cell_idx: int) -> List[int]:
        """Get successor cells (including self for staying)."""
        successors = [cell_idx]  # Can stay

        # Get neighboring cells
        grid_idx = np.unravel_index(cell_idx, self.abstraction.resolution)

        for dim in range(len(grid_idx)):
            for delta in [-1, 1]:
                neighbor = list(grid_idx)
                neighbor[dim] += delta

                # Check bounds
                if 0 <= neighbor[dim] < self.abstraction.resolution[dim]:
                    neighbor_idx = np.ravel_multi_index(neighbor, self.abstraction.resolution)
                    if neighbor_idx in self.abstraction.safe_cells:
                        successors.append(neighbor_idx)

        return successors

    def find_accepting_cycle(self, start_cell: int) -> Optional[List[int]]:
        """
        Find an accepting cycle from a starting cell.

        Uses nested DFS (Tarjan's algorithm) to find accepting cycles.

        Args:
            start_cell: Starting cell index

        Returns:
            List of cell indices forming a cycle, or None if no cycle exists
        """
        # Find initial product states
        props = self.labeling(start_cell)
        initial_buchi = list(self.buchi.initial_states)[0]
        initial_state = (start_cell, initial_buchi)

        if initial_state not in self.product_states:
            return None

        # Nested DFS for accepting cycle
        accepting_states = {
            (cell, b_state) for (cell, b_state) in self.product_states
            if b_state in self.buchi.accepting_states
        }

        # First DFS to find reachable accepting states
        visited = set()
        path = []
        parent = {}

        def dfs1(state, path_so_far):
            visited.add(state)
            path_so_far.append(state)

            for succ in self.transitions.get(state, []):
                if succ not in visited:
                    parent[succ] = state
                    result = dfs1(succ, path_so_far)
                    if result is not None:
                        return result

            # If accepting, do second DFS to find cycle back
            if state in accepting_states:
                cycle = self._find_cycle_from(state, state)
                if cycle is not None:
                    return path_so_far + cycle

            path_so_far.pop()
            return None

        result = dfs1(initial_state, [])

        if result is not None:
            # Extract cell indices from product states
            return [s[0] for s in result]

        return None

    def _find_cycle_from(self, start: Tuple[int, str], target: Tuple[int, str]) -> Optional[List[Tuple[int, str]]]:
        """Find a path from start that returns to target."""
        visited = set()
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)

            for succ in self.transitions.get(current, []):
                if succ == target and len(path) > 1:
                    return path[1:]  # Exclude start (it's already in prefix)

                if succ not in visited:
                    visited.add(succ)
                    queue.append((succ, path + [succ]))

        return None


class RecurrenceController:
    """
    Controller for recurrence/patrolling specifications.

    Computes a path that visits target regions infinitely often.
    """

    def __init__(self, abstraction, regions: Dict[str, np.ndarray],
                 spec_type: str = 'patrol', kp: float = 0.8, name: str = "Recurrence"):
        """
        Initialize recurrence controller.

        Args:
            abstraction: Grid abstraction of the workspace
            regions: Dictionary mapping region names to polygon vertices
            spec_type: 'patrol' for GF(r1) & GF(r2) & ..., 'sequence' for ordered
            kp: Proportional gain for low-level control
            name: Controller name
        """
        self.abstraction = abstraction
        self.regions = regions
        self.spec_type = spec_type
        self.kp = kp
        self.name = name

        self.model = None
        self.cycle: List[int] = []
        self.waypoint_idx = 0
        self.current_waypoint = None
        self.waypoint_tolerance = 0.5

        # Build specification automaton
        region_names = list(regions.keys())
        if spec_type == 'patrol':
            self.buchi = PatrolSpecification(region_names)
        else:
            self.buchi = SequenceSpecification(region_names)

        # Register regions with abstraction
        self._setup_regions()

    def _setup_regions(self):
        """Set up region labeling for cells."""
        self.cell_labels: Dict[int, Dict[str, bool]] = {}

        for cell_idx in self.abstraction.safe_cells:
            center = self.abstraction.get_cell_center(cell_idx)
            labels = {}

            for region_name, polygon in self.regions.items():
                labels[region_name] = self._point_in_polygon(center, polygon)

            self.cell_labels[cell_idx] = labels

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if point is inside polygon using ray casting."""
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            if ((polygon[i, 1] > point[1]) != (polygon[j, 1] > point[1]) and
                point[0] < (polygon[j, 0] - polygon[i, 0]) *
                (point[1] - polygon[i, 1]) / (polygon[j, 1] - polygon[i, 1]) + polygon[i, 0]):
                inside = not inside
            j = i

        return inside

    def _labeling_function(self, cell_idx: int) -> Dict[str, bool]:
        """Get labels for a cell."""
        return self.cell_labels.get(cell_idx, {r: False for r in self.regions})

    def set_model(self, model):
        """Set the robot model."""
        self.model = model

    def compute_patrol_cycle(self, start_state: np.ndarray) -> bool:
        """
        Compute a patrol cycle from the starting state.

        Args:
            start_state: Initial robot state

        Returns:
            True if valid cycle found
        """
        # Get starting cell
        start_cell = self.abstraction.get_cell_index(start_state[:2])
        if start_cell is None or start_cell not in self.abstraction.safe_cells:
            return False

        # Build product automaton
        product = ProductAutomaton(self.abstraction, self.buchi, self._labeling_function)

        # Find accepting cycle
        cycle = product.find_accepting_cycle(start_cell)

        if cycle is not None:
            self.cycle = cycle
            self.waypoint_idx = 0
            self._update_waypoint()
            return True

        # Fallback: compute simple cycle visiting all regions
        return self._compute_simple_cycle(start_cell)

    def _compute_simple_cycle(self, start_cell: int) -> bool:
        """Compute a simple cycle visiting all regions using shortest paths."""
        # Find cells in each region
        region_cells = {name: [] for name in self.regions}

        for cell_idx, labels in self.cell_labels.items():
            for region_name, in_region in labels.items():
                if in_region:
                    region_cells[region_name].append(cell_idx)

        # Check all regions are reachable
        for name, cells in region_cells.items():
            if not cells:
                print(f"Warning: No cells in region {name}")
                return False

        # Build cycle: start -> region1 -> region2 -> ... -> start
        cycle = [start_cell]
        current = start_cell

        for region_name in self.regions:
            # Find closest cell in this region
            target = min(region_cells[region_name],
                        key=lambda c: self._cell_distance(current, c))

            # Find path to target
            path = self._shortest_path(current, target)
            if path is None:
                return False

            cycle.extend(path[1:])  # Exclude current
            current = target

        # Close the cycle
        path_back = self._shortest_path(current, start_cell)
        if path_back is not None:
            cycle.extend(path_back[1:])

        self.cycle = cycle
        self.waypoint_idx = 0
        self._update_waypoint()
        return True

    def _cell_distance(self, cell1: int, cell2: int) -> float:
        """Compute distance between cell centers."""
        c1 = self.abstraction.get_cell_center(cell1)
        c2 = self.abstraction.get_cell_center(cell2)
        return np.linalg.norm(c1 - c2)

    def _shortest_path(self, start: int, goal: int) -> Optional[List[int]]:
        """Find shortest path between cells using Dijkstra."""
        if start == goal:
            return [start]

        distances = {start: 0}
        parent = {start: None}
        heap = [(0, start)]
        visited = set()

        while heap:
            dist, current = heapq.heappop(heap)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return path[::-1]

            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue

                new_dist = dist + self._cell_distance(current, neighbor)
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    parent[neighbor] = current
                    heapq.heappush(heap, (new_dist, neighbor))

        return None

    def _get_neighbors(self, cell_idx: int) -> List[int]:
        """Get neighboring safe cells."""
        neighbors = []
        grid_idx = np.unravel_index(cell_idx, self.abstraction.resolution)

        for dim in range(len(grid_idx)):
            for delta in [-1, 1]:
                neighbor = list(grid_idx)
                neighbor[dim] += delta

                if 0 <= neighbor[dim] < self.abstraction.resolution[dim]:
                    neighbor_idx = np.ravel_multi_index(neighbor, self.abstraction.resolution)
                    if neighbor_idx in self.abstraction.safe_cells:
                        neighbors.append(neighbor_idx)

        return neighbors

    def _update_waypoint(self):
        """Update current waypoint from cycle."""
        if self.cycle:
            cell_idx = self.cycle[self.waypoint_idx % len(self.cycle)]
            self.current_waypoint = self.abstraction.get_cell_center(cell_idx)

    def compute_control(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, Dict]:
        """
        Compute control input for patrolling.

        Args:
            x: Current state
            t: Current time

        Returns:
            Control input and diagnostic info
        """
        if self.current_waypoint is None or self.model is None:
            return np.zeros(self.model.n_inputs if self.model else 2), {}

        # Check if reached current waypoint
        pos = x[:2]
        dist = np.linalg.norm(pos - self.current_waypoint)

        if dist < self.waypoint_tolerance:
            self.waypoint_idx += 1
            self._update_waypoint()

        # Simple proportional control toward waypoint
        error = self.current_waypoint - pos
        u = self.kp * error

        # Saturate
        if self.model is not None:
            u = np.clip(u, self.model.u_bounds[:, 0], self.model.u_bounds[:, 1])

        diag = {
            'waypoint_idx': self.waypoint_idx % len(self.cycle) if self.cycle else 0,
            'distance_to_waypoint': dist,
            'current_waypoint': self.current_waypoint.copy(),
            'cycle_length': len(self.cycle)
        }

        return u, diag

    def get_patrol_info(self) -> Dict:
        """Get information about the patrol cycle."""
        return {
            'cycle_length': len(self.cycle),
            'regions': list(self.regions.keys()),
            'spec_type': self.spec_type,
            'current_waypoint_idx': self.waypoint_idx % len(self.cycle) if self.cycle else 0
        }


def create_patrol_regions(bounds: np.ndarray, n_regions: int = 4,
                          margin: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Create patrol regions within workspace bounds.

    Args:
        bounds: Workspace bounds [[x_min, x_max], [y_min, y_max]]
        n_regions: Number of patrol regions
        margin: Margin from workspace boundary

    Returns:
        Dictionary of region name -> polygon vertices
    """
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Add margin
    x_min += margin
    x_max -= margin
    y_min += margin
    y_max -= margin

    regions = {}

    if n_regions == 4:
        # Four corners
        region_size = 1.5

        regions['NE'] = np.array([
            [x_max - region_size, y_max - region_size],
            [x_max, y_max - region_size],
            [x_max, y_max],
            [x_max - region_size, y_max]
        ])

        regions['NW'] = np.array([
            [x_min, y_max - region_size],
            [x_min + region_size, y_max - region_size],
            [x_min + region_size, y_max],
            [x_min, y_max]
        ])

        regions['SW'] = np.array([
            [x_min, y_min],
            [x_min + region_size, y_min],
            [x_min + region_size, y_min + region_size],
            [x_min, y_min + region_size]
        ])

        regions['SE'] = np.array([
            [x_max - region_size, y_min],
            [x_max, y_min],
            [x_max, y_min + region_size],
            [x_max - region_size, y_min + region_size]
        ])

    elif n_regions == 2:
        # Two opposite corners
        region_size = 2.0

        regions['A'] = np.array([
            [x_min, y_min],
            [x_min + region_size, y_min],
            [x_min + region_size, y_min + region_size],
            [x_min, y_min + region_size]
        ])

        regions['B'] = np.array([
            [x_max - region_size, y_max - region_size],
            [x_max, y_max - region_size],
            [x_max, y_max],
            [x_max - region_size, y_max]
        ])

    else:
        # Distribute evenly
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        radius = min(x_max - cx, y_max - cy) * 0.7
        region_size = 1.0

        for i in range(n_regions):
            angle = 2 * np.pi * i / n_regions
            rx = cx + radius * np.cos(angle)
            ry = cy + radius * np.sin(angle)

            regions[f'R{i+1}'] = np.array([
                [rx - region_size/2, ry - region_size/2],
                [rx + region_size/2, ry - region_size/2],
                [rx + region_size/2, ry + region_size/2],
                [rx - region_size/2, ry + region_size/2]
            ])

    return regions
