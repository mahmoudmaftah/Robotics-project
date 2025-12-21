"""
Optimized Abstraction Module with Parallel Processing.

This module provides optimized versions of the abstraction operations:
- VectorizedAbstraction: Uses NumPy vectorization and pre-computed bounds
- ParallelAbstraction: Uses multiprocessing for parallel computation
- GPUAbstraction: Uses CuPy for GPU acceleration (batch processing)

Optional CuPy support for GPU acceleration (if available).
"""

import numpy as np
from typing import Dict, Set, Tuple, Optional, List
from itertools import product
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

# Try to import CuPy
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = np
    HAS_CUPY = False

from .dynamics import Dynamics


# Global variables for worker processes (to avoid pickling dynamics)
_worker_dynamics = None
_worker_state_bounds = None
_worker_eta = None
_worker_grid_shape = None


def _init_worker(dynamics, state_bounds, eta, grid_shape):
    """Initialize worker process with shared data."""
    global _worker_dynamics, _worker_state_bounds, _worker_eta, _worker_grid_shape
    _worker_dynamics = dynamics
    _worker_state_bounds = state_bounds
    _worker_eta = eta
    _worker_grid_shape = grid_shape


def _compute_cell_transitions(args):
    """Worker function for parallel transition computation (single cell)."""
    cell_idx, x_lo, x_hi = args

    global _worker_dynamics, _worker_state_bounds, _worker_eta, _worker_grid_shape
    dynamics = _worker_dynamics
    state_bounds = _worker_state_bounds
    eta = _worker_eta
    grid_shape = _worker_grid_shape

    results = {}
    state_dim = len(x_lo)

    for u_idx, u in enumerate(dynamics.control_set):
        # Compute over-approximated successor
        succ_lo, succ_hi = dynamics.post(x_lo, x_hi, u)

        # Check if outside grid bounds
        out_of_bounds = False
        for d in range(state_dim):
            if succ_lo[d] < state_bounds[d, 0] or succ_hi[d] > state_bounds[d, 1]:
                out_of_bounds = True
                break

        if out_of_bounds:
            results[(cell_idx, u_idx)] = set()
            continue

        # Find overlapping cells
        cells = set()
        idx_ranges = []
        valid = True

        for d in range(state_dim):
            lo_clamped = max(succ_lo[d], state_bounds[d, 0])
            hi_clamped = min(succ_hi[d], state_bounds[d, 1])

            if lo_clamped >= hi_clamped:
                valid = False
                break

            idx_lo = max(0, int((lo_clamped - state_bounds[d, 0]) / eta))
            idx_hi = min(grid_shape[d], int(np.ceil((hi_clamped - state_bounds[d, 0]) / eta)))
            idx_ranges.append(range(idx_lo, idx_hi))

        if valid:
            for multi_idx in product(*idx_ranges):
                flat_idx = np.ravel_multi_index(multi_idx, grid_shape)
                cells.add(flat_idx)

        results[(cell_idx, u_idx)] = cells

    return results


class GPUAbstraction:
    """
    Parallel finite-state abstraction using threading.

    The dynamics computation is CPU-bound and cannot be easily GPU-accelerated,
    so this class uses ThreadPoolExecutor for parallel computation.
    CuPy is used for vectorized bound storage if available.
    """

    def __init__(self, dynamics: Dynamics, state_bounds: np.ndarray, eta: float, n_workers: int = None):
        """
        Args:
            dynamics: A Dynamics object (e.g., IntegratorDynamics)
            state_bounds: Array of shape (state_dim, 2) with [min, max] per dimension
            eta: Grid cell size (same for all dimensions)
            n_workers: Number of worker threads (default: CPU count)
        """
        self.dynamics = dynamics
        self.state_bounds = np.array(state_bounds)
        self.eta = eta
        self.n_workers = n_workers or mp.cpu_count()

        # Compute grid dimensions
        self.grid_shape = tuple(
            int(np.ceil((bounds[1] - bounds[0]) / eta))
            for bounds in self.state_bounds
        )
        self.num_cells = int(np.prod(self.grid_shape))

        # Transitions: {(cell_idx, u_idx): set of successor cell indices}
        self.transitions: Dict[Tuple[int, int], Set[int]] = {}

        # Pre-compute cell bounds
        self._cell_bounds_lo = None
        self._cell_bounds_hi = None

        print(f"Parallel abstraction with {self.n_workers} workers")
        if HAS_CUPY:
            print(f"  CuPy {cp.__version__} available for vectorized operations")

    def _precompute_cell_bounds(self) -> None:
        """Pre-compute bounds for all cells (vectorized)."""
        state_dim = self.dynamics.state_dim

        # Use NumPy meshgrid for efficient index generation
        ranges = [np.arange(s) for s in self.grid_shape]
        grids = np.meshgrid(*ranges, indexing='ij')
        all_indices = np.stack([g.ravel() for g in grids], axis=1)

        # Compute bounds for all cells at once
        self._cell_bounds_lo = np.zeros((self.num_cells, state_dim))
        self._cell_bounds_hi = np.zeros((self.num_cells, state_dim))

        for d in range(state_dim):
            self._cell_bounds_lo[:, d] = self.state_bounds[d, 0] + all_indices[:, d] * self.eta
            self._cell_bounds_hi[:, d] = self._cell_bounds_lo[:, d] + self.eta

    def cell_to_bounds(self, cell_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a flat cell index to its spatial bounds."""
        if self._cell_bounds_lo is not None:
            return self._cell_bounds_lo[cell_idx].copy(), self._cell_bounds_hi[cell_idx].copy()

        multi_idx = np.unravel_index(cell_idx, self.grid_shape)
        lo = np.array([
            self.state_bounds[d, 0] + multi_idx[d] * self.eta
            for d in range(self.dynamics.state_dim)
        ])
        hi = lo + self.eta
        return lo, hi

    def bounds_to_cells(self, lo: np.ndarray, hi: np.ndarray) -> Set[int]:
        """Find all cell indices that overlap with the rectangle [lo, hi]."""
        cells = set()
        state_dim = self.dynamics.state_dim

        # Check if outside grid bounds
        for d in range(state_dim):
            if lo[d] < self.state_bounds[d, 0] or hi[d] > self.state_bounds[d, 1]:
                return set()

        # Compute index ranges for each dimension
        idx_ranges = []
        for d in range(state_dim):
            lo_clamped = max(lo[d], self.state_bounds[d, 0])
            hi_clamped = min(hi[d], self.state_bounds[d, 1])

            if lo_clamped >= hi_clamped:
                return set()

            idx_lo = max(0, int((lo_clamped - self.state_bounds[d, 0]) / self.eta))
            idx_hi = min(self.grid_shape[d], int(np.ceil((hi_clamped - self.state_bounds[d, 0]) / self.eta)))
            idx_ranges.append(range(idx_lo, idx_hi))

        # Generate all combinations
        for multi_idx in product(*idx_ranges):
            flat_idx = np.ravel_multi_index(multi_idx, self.grid_shape)
            cells.add(flat_idx)

        return cells

    def _compute_cell_transitions(self, cell_idx: int) -> Dict[Tuple[int, int], Set[int]]:
        """Compute transitions for a single cell (all controls)."""
        results = {}
        x_lo, x_hi = self.cell_to_bounds(cell_idx)

        for u_idx, u in enumerate(self.dynamics.control_set):
            succ_lo, succ_hi = self.dynamics.post(x_lo, x_hi, u)
            succ_cells = self.bounds_to_cells(succ_lo, succ_hi)
            results[(cell_idx, u_idx)] = succ_cells

        return results

    def build_transitions(self, use_parallel: bool = True) -> None:
        """
        Compute all transitions for every (cell, control) pair.

        Args:
            use_parallel: If True, use parallel processing
        """
        # Pre-compute cell bounds
        if self._cell_bounds_lo is None:
            self._precompute_cell_bounds()

        if use_parallel and self.num_cells > 100:
            self._build_transitions_parallel()
        else:
            self._build_transitions_sequential()

    def _build_transitions_parallel(self) -> None:
        """Build transitions using ThreadPoolExecutor."""
        print(f"Building transitions in parallel ({self.n_workers} workers) for {self.num_cells} cells × {len(self.dynamics.control_set)} controls...")

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all cells
            futures = []
            for cell_idx in range(self.num_cells):
                future = executor.submit(self._compute_cell_transitions, cell_idx)
                futures.append(future)

            # Collect results with progress bar
            for future in tqdm(futures, desc="Cells"):
                cell_results = future.result()
                self.transitions.update(cell_results)

        print(f"Done. Total transitions: {len(self.transitions)}")

    def _build_transitions_sequential(self) -> None:
        """Build transitions sequentially."""
        print(f"Building transitions for {self.num_cells} cells × {len(self.dynamics.control_set)} controls...")

        for cell_idx in tqdm(range(self.num_cells)):
            results = self._compute_cell_transitions(cell_idx)
            self.transitions.update(results)

        print(f"Done. Total transitions: {len(self.transitions)}")

    def get_cell_center(self, cell_idx: int) -> np.ndarray:
        """Get the center point of a cell."""
        lo, hi = self.cell_to_bounds(cell_idx)
        return (lo + hi) / 2

    def point_to_cell(self, x: np.ndarray) -> int:
        """Convert a continuous state to its cell index. Returns -1 if out of bounds."""
        multi_idx = []
        for d in range(self.dynamics.state_dim):
            idx = int((x[d] - self.state_bounds[d, 0]) / self.eta)
            if idx < 0 or idx >= self.grid_shape[d]:
                return -1
            multi_idx.append(idx)
        return np.ravel_multi_index(tuple(multi_idx), self.grid_shape)

    def is_in_bounds(self, x: np.ndarray) -> bool:
        """Check if a state is within the grid bounds."""
        for d in range(self.dynamics.state_dim):
            if x[d] < self.state_bounds[d, 0] or x[d] >= self.state_bounds[d, 1]:
                return False
        return True


class VectorizedAbstraction:
    """
    NumPy-vectorized abstraction for systems with simple dynamics (e.g., Integrator).

    Uses vectorized NumPy operations for faster computation without GPU.
    Best for systems where dynamics.post() can be vectorized.
    """

    def __init__(self, dynamics: Dynamics, state_bounds: np.ndarray, eta: float):
        self.dynamics = dynamics
        self.state_bounds = np.array(state_bounds)
        self.eta = eta

        self.grid_shape = tuple(
            int(np.ceil((bounds[1] - bounds[0]) / eta))
            for bounds in self.state_bounds
        )
        self.num_cells = int(np.prod(self.grid_shape))
        self.transitions: Dict[Tuple[int, int], Set[int]] = {}

        # Pre-compute all cell bounds
        self._precompute_cell_bounds()

    def _precompute_cell_bounds(self) -> None:
        """Pre-compute bounds for all cells using vectorized operations."""
        state_dim = self.dynamics.state_dim
        ranges = [np.arange(s) for s in self.grid_shape]
        grids = np.meshgrid(*ranges, indexing='ij')

        # Flatten and stack indices
        all_indices = np.stack([g.ravel() for g in grids], axis=1)  # (num_cells, state_dim)

        # Vectorized computation of bounds
        self._cell_bounds_lo = np.zeros((self.num_cells, state_dim))
        self._cell_bounds_hi = np.zeros((self.num_cells, state_dim))

        for d in range(state_dim):
            self._cell_bounds_lo[:, d] = self.state_bounds[d, 0] + all_indices[:, d] * self.eta
            self._cell_bounds_hi[:, d] = self._cell_bounds_lo[:, d] + self.eta

    def cell_to_bounds(self, cell_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounds for a single cell."""
        return self._cell_bounds_lo[cell_idx], self._cell_bounds_hi[cell_idx]

    def bounds_to_cells(self, lo: np.ndarray, hi: np.ndarray) -> Set[int]:
        """Find all cell indices that overlap with the rectangle [lo, hi]."""
        cells = set()
        state_dim = self.dynamics.state_dim

        for d in range(state_dim):
            if lo[d] < self.state_bounds[d, 0] or hi[d] > self.state_bounds[d, 1]:
                return set()

        idx_ranges = []
        for d in range(state_dim):
            lo_clamped = max(lo[d], self.state_bounds[d, 0])
            hi_clamped = min(hi[d], self.state_bounds[d, 1])

            if lo_clamped >= hi_clamped:
                return set()

            idx_lo = max(0, int((lo_clamped - self.state_bounds[d, 0]) / self.eta))
            idx_hi = min(self.grid_shape[d], int(np.ceil((hi_clamped - self.state_bounds[d, 0]) / self.eta)))
            idx_ranges.append(range(idx_lo, idx_hi))

        for multi_idx in product(*idx_ranges):
            flat_idx = np.ravel_multi_index(multi_idx, self.grid_shape)
            cells.add(flat_idx)

        return cells

    def build_transitions(self) -> None:
        """Build transitions using vectorized operations where possible."""
        print(f"Building transitions (vectorized) for {self.num_cells} cells × {len(self.dynamics.control_set)} controls...")

        for cell_idx in tqdm(range(self.num_cells)):
            x_lo = self._cell_bounds_lo[cell_idx]
            x_hi = self._cell_bounds_hi[cell_idx]

            for u_idx, u in enumerate(self.dynamics.control_set):
                succ_lo, succ_hi = self.dynamics.post(x_lo, x_hi, u)
                succ_cells = self.bounds_to_cells(succ_lo, succ_hi)
                self.transitions[(cell_idx, u_idx)] = succ_cells

        print(f"Done. Total transitions: {len(self.transitions)}")

    def get_cell_center(self, cell_idx: int) -> np.ndarray:
        return (self._cell_bounds_lo[cell_idx] + self._cell_bounds_hi[cell_idx]) / 2

    def point_to_cell(self, x: np.ndarray) -> int:
        multi_idx = []
        for d in range(self.dynamics.state_dim):
            idx = int((x[d] - self.state_bounds[d, 0]) / self.eta)
            if idx < 0 or idx >= self.grid_shape[d]:
                return -1
            multi_idx.append(idx)
        return np.ravel_multi_index(tuple(multi_idx), self.grid_shape)

    def is_in_bounds(self, x: np.ndarray) -> bool:
        for d in range(self.dynamics.state_dim):
            if x[d] < self.state_bounds[d, 0] or x[d] >= self.state_bounds[d, 1]:
                return False
        return True


def create_abstraction(
    dynamics: Dynamics,
    state_bounds: np.ndarray,
    eta: float,
    use_gpu: bool = True
):
    """
    Factory function to create the best abstraction for the current environment.

    Args:
        dynamics: Dynamics object
        state_bounds: State bounds array
        eta: Cell size
        use_gpu: If True, try to use GPU acceleration

    Returns:
        Abstraction object (GPU, vectorized, or standard)
    """
    if use_gpu and HAS_CUPY:
        return GPUAbstraction(dynamics, state_bounds, eta)
    else:
        return VectorizedAbstraction(dynamics, state_bounds, eta)
