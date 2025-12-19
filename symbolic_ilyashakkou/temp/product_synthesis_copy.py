"""
Product Automaton Synthesis - Controller synthesis for temporal specifications.

This module implements the Product Automaton approach from the paper:
1. Combine the physical system (grid abstraction) with a mission automaton (NFA)
2. Run reachability synthesis on the product to satisfy temporal specs

Product State: (ψ, ξ) where ψ = NFA state, ξ = grid cell
Dynamics: When robot moves ξ → ξ', the NFA observes the label of ξ' and updates ψ → ψ'

Usage:
    # Define regions and specification
    labeler = RegionLabeler()
    labeler.add_region("A", [[1, 2], [1, 2]])
    labeler.add_region("B", [[3, 4], [3, 4]])
    
    spec = "(A|B)C"  # Visit A or B, then visit C
    
    # Run synthesis
    synth = ProductSynthesis(abstraction, labeler, spec, obstacles)
    synth.run()
    
    # Simulate
    synth.simulate(start_position)
"""

import random
import numpy as np
from typing import Dict, Set, List, Tuple, Optional, FrozenSet
from dataclasses import dataclass

from .abstraction import Abstraction
from .nfa import NFA, RegionLabeler, regex_to_nfa


# =============================================================================
# Product Automaton
# =============================================================================

@dataclass
class ProductState:
    """
    A state in the product automaton: (NFA states, grid cell, last observed label).
    
    The last_label tracks which labeled region the robot most recently entered.
    This prevents re-emitting the same label when staying in a region.
    """
    nfa_states: FrozenSet[int]  # Set of NFA states (for NFA nondeterminism)
    cell_idx: int               # Grid cell index
    last_label: Optional[str] = None  # Last observed region label
    
    def __hash__(self):
        return hash((self.nfa_states, self.cell_idx, self.last_label))
    
    def __eq__(self, other):
        return (self.nfa_states == other.nfa_states and 
                self.cell_idx == other.cell_idx and
                self.last_label == other.last_label)


class ProductAutomaton:
    """
    Product of grid abstraction and NFA specification.
    
    States: (ψ, ξ) where ψ is a set of NFA states, ξ is a grid cell
    Transitions: Coupled physical + logical updates
    """
    
    def __init__(
        self,
        abstraction: Abstraction,
        nfa: NFA,
        labeler: RegionLabeler,
        obstacles: Set[int] = None
    ):
        """
        Args:
            abstraction: Grid abstraction with precomputed transitions
            nfa: Mission specification automaton
            labeler: Maps grid cells to symbolic labels
            obstacles: Grid cells to avoid (will be pruned from product)
        """
        self.abstraction = abstraction
        self.nfa = nfa
        self.labeler = labeler
        self.obstacles = obstacles or set()
        
        # Product states and transitions (built lazily)
        self.states: Set[ProductState] = set()
        self.transitions: Dict[Tuple[ProductState, int], Set[ProductState]] = {}
        self.initial_states: Set[ProductState] = set()
        self.accepting_states: Set[ProductState] = set()
        
        # For efficient lookup: encode product states as integers
        self._state_to_id: Dict[ProductState, int] = {}
        self._id_to_state: Dict[int, ProductState] = {}
        self._next_id = 0
    
    def _encode_state(self, state: ProductState) -> int:
        """Get unique integer ID for a product state."""
        if state not in self._state_to_id:
            self._state_to_id[state] = self._next_id
            self._id_to_state[self._next_id] = state
            self._next_id += 1
        return self._state_to_id[state]
    
    def _decode_state(self, state_id: int) -> ProductState:
        """Get product state from integer ID."""
        return self._id_to_state[state_id]
    
    def _get_cell_label(self, cell_idx: int) -> Optional[str]:
        """Get the label of a grid cell."""
        lo, hi = self.abstraction.cell_to_bounds(cell_idx)
        center = (lo + hi) / 2
        return self.labeler.get_label(center)
    
    def build(self, verbose: bool = True):
        """
        Build the product automaton by exploring reachable states.
        
        Uses BFS from initial states to discover all reachable product states.
        """
        if verbose:
            print("=" * 50)
            print("BUILDING PRODUCT AUTOMATON")
            print("=" * 50)
        
        num_controls = len(self.abstraction.dynamics.control_set)
        safe_cells = set(range(self.abstraction.num_cells)) - self.obstacles
        
        # Initial NFA states (epsilon closure of initial)
        initial_nfa_states = frozenset(self.nfa.get_initial_states())
        
        if verbose:
            print(f"NFA states: {len(self.nfa.states)}")
            print(f"NFA alphabet: {self.nfa.alphabet}")
            print(f"Initial NFA states: {initial_nfa_states}")
            print(f"Accepting NFA states: {self.nfa.accepting}")
            print(f"Safe grid cells: {len(safe_cells)}")
        
        # BFS to discover reachable product states
        queue = []
        visited = set()
        
        # Initialize: all (initial_nfa, safe_cell) pairs
        # At initialization, we observe the label of the starting cell
        for cell_idx in safe_cells:
            initial_label = self._get_cell_label(cell_idx)
            
            # Update NFA based on initial cell's label (if any)
            if initial_label is not None:
                nfa_after_label = frozenset(
                    self.nfa.get_next_states(initial_nfa_states, initial_label)
                )
                if not nfa_after_label:
                    # Can't start in this cell - NFA rejects initial label
                    continue
            else:
                nfa_after_label = initial_nfa_states
            
            state = ProductState(nfa_after_label, cell_idx, initial_label)
            self.states.add(state)
            self.initial_states.add(state)
            self._encode_state(state)
            queue.append(state)
            visited.add(state)
            
            # Check if accepting
            if self.nfa.has_accepting(nfa_after_label):
                self.accepting_states.add(state)
        
        if verbose:
            print(f"Initial product states: {len(self.initial_states)}")
        
        # BFS exploration
        transitions_count = 0
        while queue:
            current = queue.pop(0)
            
            # Skip if cell is obstacle
            if current.cell_idx in self.obstacles:
                continue
            
            # Try all controls
            for u_idx in range(num_controls):
                # Get physical successors
                succ_cells = self.abstraction.transitions.get(
                    (current.cell_idx, u_idx), set()
                )
                
                # Skip if no successors or successors go outside safe set
                if not succ_cells:
                    continue
                if not succ_cells.issubset(safe_cells):
                    continue
                
                # Compute product successors
                product_successors = set()
                for succ_cell in succ_cells:
                    # Get label of successor cell
                    label = self._get_cell_label(succ_cell)
                    
                    # KEY FIX: Only emit label if it's DIFFERENT from last_label
                    # This prevents re-observing the same label when staying in a region
                    if label != current.last_label and label is not None:
                        # Entering a new labeled region - emit label to NFA
                        next_nfa_states = frozenset(
                            self.nfa.get_next_states(current.nfa_states, label)
                        )
                        new_last_label = label
                        
                        if not next_nfa_states:
                            # NFA has no valid transition - skip this successor
                            continue
                    else:
                        # Same label or unlabeled - NFA stays in place
                        next_nfa_states = current.nfa_states
                        # Update last_label: keep current if staying in region, clear if leaving to unlabeled
                        new_last_label = label if label is not None else current.last_label
                    
                    succ_state = ProductState(next_nfa_states, succ_cell, new_last_label)
                    product_successors.add(succ_state)
                    
                    # Add to exploration queue if new
                    if succ_state not in visited:
                        visited.add(succ_state)
                        self.states.add(succ_state)
                        self._encode_state(succ_state)
                        queue.append(succ_state)
                        
                        # Check if accepting
                        if self.nfa.has_accepting(next_nfa_states):
                            self.accepting_states.add(succ_state)
                
                # Store transition if non-empty
                if product_successors:
                    self.transitions[(current, u_idx)] = product_successors
                    transitions_count += 1
        
        if verbose:
            print(f"Product states: {len(self.states)}")
            print(f"Accepting states: {len(self.accepting_states)}")
            print(f"Transitions: {transitions_count}")
            print("=" * 50)
    
    def get_successors(self, state: ProductState, u_idx: int) -> Set[ProductState]:
        """Get successor product states for a (state, control) pair."""
        return self.transitions.get((state, u_idx), set())


# =============================================================================
# Reachability on Product Automaton
# =============================================================================

def compute_product_reachability(
    product: ProductAutomaton,
    verbose: bool = True
) -> Tuple[Set[ProductState], List[Set[ProductState]], Dict[ProductState, List[int]]]:
    """
    Compute reachability to accepting states on the product automaton.
    
    Algorithm (Reachability Fixed-Point):
        R₀ = accepting states
        Rₖ₊₁ = R₀ ∪ Pre(Rₖ)
        Stop when Rₖ₊₁ = Rₖ
    
    Returns:
        (winning_set, reach_sets, controller)
    """
    if verbose:
        print("=" * 50)
        print("REACHABILITY ON PRODUCT AUTOMATON")
        print("=" * 50)
    
    target = product.accepting_states
    
    if not target:
        if verbose:
            print("WARNING: No accepting states!")
        return set(), [], {}
    
    num_controls = len(product.abstraction.dynamics.control_set)
    
    # Fixed-point iteration
    r_k = target.copy()
    reach_sets = [r_k.copy()]
    
    if verbose:
        print(f"R₀ (accepting): {len(r_k)} states")
    
    iteration = 0
    while True:
        iteration += 1
        
        # Pre(Rₖ): states that can reach Rₖ in one step
        pre_r_k = set()
        for state in product.states:
            for u_idx in range(num_controls):
                successors = product.get_successors(state, u_idx)
                # Check if ALL successors are in Rₖ (robust Pre)
                if successors and successors.issubset(r_k):
                    pre_r_k.add(state)
                    break
        
        # Rₖ₊₁ = R₀ ∪ Pre(Rₖ)
        r_k_plus_1 = target | pre_r_k
        added = len(r_k_plus_1) - len(r_k)
        
        if verbose:
            print(f"Iteration {iteration}: {len(r_k_plus_1)} states (+{added})")
        
        reach_sets.append(r_k_plus_1.copy())
        
        if r_k_plus_1 == r_k:
            break
        r_k = r_k_plus_1
    
    winning_set = r_k
    
    if verbose:
        print(f"Converged! Winning set: {len(winning_set)} states")
    
    # Extract controller
    controller = _extract_product_controller(product, target, reach_sets)
    
    if verbose:
        print(f"Controller covers {len(controller)} states")
        print("=" * 50)
    
    return winning_set, reach_sets, controller


def _extract_product_controller(
    product: ProductAutomaton,
    target: Set[ProductState],
    reach_sets: List[Set[ProductState]]
) -> Dict[ProductState, List[int]]:
    """Extract controller from reachability layers on product."""
    controller = {}
    num_controls = len(product.abstraction.dynamics.control_set)
    winning_set = reach_sets[-1] if reach_sets else set()
    
    for state in winning_set:
        # Find layer
        layer = -1
        for k, r_k in enumerate(reach_sets):
            if state in r_k:
                layer = k
                break
        
        if layer == 0:
            # At accepting: any control that stays in winning set
            valid = []
            for u_idx in range(num_controls):
                succ = product.get_successors(state, u_idx)
                if succ and succ.issubset(winning_set):
                    valid.append(u_idx)
        else:
            # Not at accepting: controls that reach layer k-1
            target_layer = reach_sets[layer - 1]
            valid = []
            for u_idx in range(num_controls):
                succ = product.get_successors(state, u_idx)
                if succ and succ.issubset(target_layer):
                    valid.append(u_idx)
        
        if valid:
            controller[state] = valid
    
    return controller


# =============================================================================
# Main Synthesis Class
# =============================================================================

class ProductSynthesis:
    """
    Temporal Logic Synthesis using Product Automaton.
    
    Given a regex specification (converted to NFA), synthesize a controller
    that guides the robot to satisfy the specification while avoiding obstacles.
    
    Usage:
        labeler = RegionLabeler()
        labeler.add_region("A", [[1, 2], [1, 2]])
        labeler.add_region("B", [[3, 4], [3, 4]])
        
        synth = ProductSynthesis(abstraction, labeler, "(A|B)C", obstacles)
        synth.run()
        trajectory = synth.simulate(start_pos)
    """
    
    def __init__(
        self,
        abstraction: Abstraction,
        labeler: RegionLabeler,
        spec: str,
        obstacles: Set[int] = None
    ):
        """
        Args:
            abstraction: Grid abstraction with precomputed transitions
            labeler: Maps grid cells to symbolic labels
            spec: Regular expression specification (e.g., "AB", "(A|B)C*")
            obstacles: Grid cells to avoid
        """
        self.abstraction = abstraction
        self.labeler = labeler
        self.spec = spec
        self.obstacles = obstacles or set()
        
        # Convert spec to NFA
        self.nfa = regex_to_nfa(spec)
        
        # Build product automaton
        self.product = ProductAutomaton(abstraction, self.nfa, labeler, self.obstacles)
        
        # Results (populated by run())
        self.winning_set: Set[ProductState] = set()
        self.reach_sets: List[Set[ProductState]] = []
        self.controller: Dict[ProductState, List[int]] = {}
    
    def run(self, verbose: bool = True) -> Set[ProductState]:
        """
        Run the full synthesis pipeline.
        
        Returns:
            Winning set of product states
        """
        if verbose:
            print(f"Specification: {self.spec}")
            print(f"Obstacles: {len(self.obstacles)} cells")
        
        # Build product automaton
        self.product.build(verbose=verbose)
        
        if not self.product.accepting_states:
            if verbose:
                print("ERROR: No accepting states in product!")
            return set()
        
        # Run reachability
        self.winning_set, self.reach_sets, self.controller = compute_product_reachability(
            self.product, verbose=verbose
        )
        
        return self.winning_set
    
    def get_control(
        self,
        nfa_states: FrozenSet[int],
        cell_idx: int,
        last_label: Optional[str] = None,
        random_choice: bool = True
    ) -> Optional[int]:
        """Get control for a product state."""
        state = ProductState(nfa_states, cell_idx, last_label)
        if state not in self.controller:
            return None
        controls = self.controller[state]
        return random.choice(controls) if random_choice else controls[0]
    
    def get_initial_nfa_states(self) -> FrozenSet[int]:
        """Get the initial NFA states (epsilon closure)."""
        return frozenset(self.nfa.get_initial_states())
    
    def update_nfa_states(
        self,
        current_nfa_states: FrozenSet[int],
        cell_idx: int,
        last_label: Optional[str] = None
    ) -> Tuple[FrozenSet[int], Optional[str]]:
        """
        Update NFA states based on the label of a cell.
        
        Only emits label if it's different from last_label (entering new region).
        
        Returns:
            (new_nfa_states, new_last_label)
        """
        label = self.product._get_cell_label(cell_idx)
        
        if label != last_label and label is not None:
            # Entering a new labeled region - emit to NFA
            new_nfa_states = frozenset(self.nfa.get_next_states(current_nfa_states, label))
            new_last_label = label
        else:
            # Same label or unlabeled - NFA stays in place
            new_nfa_states = current_nfa_states
            new_last_label = label if label is not None else last_label
        
        return new_nfa_states, new_last_label
    
    def is_accepting(self, nfa_states: FrozenSet[int]) -> bool:
        """Check if current NFA states include an accepting state."""
        return self.nfa.has_accepting(nfa_states)
    
    def is_winning(self, nfa_states: FrozenSet[int], cell_idx: int, last_label: Optional[str] = None) -> bool:
        """Check if a product state is in the winning set."""
        state = ProductState(nfa_states, cell_idx, last_label)
        return state in self.winning_set
    
    def get_distance_to_accepting(self, nfa_states: FrozenSet[int], cell_idx: int, last_label: Optional[str] = None) -> int:
        """Get minimum steps to reach accepting state, or -1 if unreachable."""
        state = ProductState(nfa_states, cell_idx, last_label)
        for k, r_k in enumerate(self.reach_sets):
            if state in r_k:
                return k
        return -1
    
    def simulate(
        self,
        start_pos: np.ndarray,
        max_steps: int = 100,
        verbose: bool = True
    ) -> Tuple[np.ndarray, List[FrozenSet[int]]]:
        """
        Simulate the controller from a starting position.
        
        Returns:
            (trajectory, nfa_history) - physical trajectory and NFA state history
        """
        trajectory = [np.array(start_pos)]
        pos = np.array(start_pos)
        
        # Initialize NFA states
        nfa_states = self.get_initial_nfa_states()
        nfa_history = [nfa_states]
        
        # Get initial cell
        cell_idx = self.abstraction.point_to_cell(pos)
        if cell_idx == -1:
            if verbose:
                print("Start position out of bounds!")
            return np.array(trajectory), nfa_history
        
        # Initialize last_label and update NFA for initial position
        last_label = None
        nfa_states, last_label = self.update_nfa_states(nfa_states, cell_idx, last_label)
        nfa_history[-1] = nfa_states
        
        if verbose:
            label = self.product._get_cell_label(cell_idx)
            print(f"Start: cell {cell_idx}, label={label}, NFA={nfa_states}")
            print(f"Accepting: {self.is_accepting(nfa_states)}")
        
        for step in range(max_steps):
            # Check if we've reached accepting state
            if self.is_accepting(nfa_states):
                if verbose:
                    print(f"Step {step}: MISSION COMPLETE! Reached accepting state.")
                break
            
            # Get control (now includes last_label)
            u_idx = self.get_control(nfa_states, cell_idx, last_label)
            if u_idx is None:
                if verbose:
                    print(f"Step {step}: No control available (not in winning set)")
                break
            
            u = self.abstraction.dynamics.control_set[u_idx]
            
            # Apply dynamics with random disturbance
            w = np.random.uniform(
                self.abstraction.dynamics.w_min,
                self.abstraction.dynamics.w_max,
                size=self.abstraction.dynamics.state_dim
            )
            next_pos = self.abstraction.dynamics.step(pos, u, w)
            
            # Update cell
            next_cell = self.abstraction.point_to_cell(next_pos)
            if next_cell == -1:
                if verbose:
                    print(f"Step {step}: Out of bounds!")
                break
            
            # Update NFA states based on label change
            next_nfa_states, next_last_label = self.update_nfa_states(nfa_states, next_cell, last_label)
            
            if verbose:
                label = self.product._get_cell_label(next_cell)
                label_changed = label != last_label and label is not None
                print(f"Step {step}: cell {cell_idx}→{next_cell}, label={label}"
                      f"{' (new!)' if label_changed else ''}, "
                      f"NFA={nfa_states}→{next_nfa_states}")
            
            trajectory.append(next_pos.copy())
            nfa_history.append(next_nfa_states)
            
            pos = next_pos
            cell_idx = next_cell
            nfa_states = next_nfa_states
            last_label = next_last_label
        
        return np.array(trajectory), nfa_history
    
    def get_winning_cells_for_nfa_state(self, nfa_states: FrozenSet[int]) -> Set[int]:
        """Get all grid cells that are winning for a given NFA state."""
        cells = set()
        for state in self.winning_set:
            if state.nfa_states == nfa_states:
                cells.add(state.cell_idx)
        return cells
