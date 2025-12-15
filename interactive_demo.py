#!/usr/bin/env python3
"""
Interactive Demo for Reach-Avoid Planning.

Features:
- User input for configuration parameters
- Random obstacle and goal region generation
- Random initial states for each run
- Multiple controller comparison
- Real-time trajectory visualization

Usage:
    python interactive_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
from matplotlib.collections import PatchCollection
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.integrator import IntegratorModel
from controllers.proportional import ProportionalController
from controllers.lqr import LQRController
from symbolic.grid_abstraction import GridAbstraction
from symbolic.reach_avoid import ReachAvoidPlanner, ReachAvoidController
from sim.simulator import Simulator


def get_user_input():
    """Get configuration from user input."""
    print("\n" + "=" * 60)
    print("INTERACTIVE REACH-AVOID DEMO")
    print("=" * 60)

    print("\nConfiguration Options:")
    print("-" * 40)

    # Number of obstacles
    while True:
        try:
            n_obstacles = input("Number of obstacles (1-5) [default=3]: ").strip()
            n_obstacles = int(n_obstacles) if n_obstacles else 3
            if 1 <= n_obstacles <= 5:
                break
            print("Please enter a number between 1 and 5")
        except ValueError:
            print("Invalid input. Using default (3)")
            n_obstacles = 3
            break

    # Grid resolution
    while True:
        try:
            resolution = input("Grid resolution (10-50) [default=20]: ").strip()
            resolution = int(resolution) if resolution else 20
            if 10 <= resolution <= 50:
                break
            print("Please enter a number between 10 and 50")
        except ValueError:
            print("Invalid input. Using default (20)")
            resolution = 20
            break

    # Number of random runs
    while True:
        try:
            n_runs = input("Number of random runs (1-10) [default=5]: ").strip()
            n_runs = int(n_runs) if n_runs else 5
            if 1 <= n_runs <= 10:
                break
            print("Please enter a number between 1 and 10")
        except ValueError:
            print("Invalid input. Using default (5)")
            n_runs = 5
            break

    # Simulation time
    while True:
        try:
            sim_time = input("Simulation time in seconds (5-30) [default=15]: ").strip()
            sim_time = float(sim_time) if sim_time else 15.0
            if 5 <= sim_time <= 30:
                break
            print("Please enter a number between 5 and 30")
        except ValueError:
            print("Invalid input. Using default (15)")
            sim_time = 15.0
            break

    # Seed option
    seed_input = input("Random seed (press Enter for random): ").strip()
    seed = int(seed_input) if seed_input else None

    return {
        'n_obstacles': n_obstacles,
        'resolution': resolution,
        'n_runs': n_runs,
        'sim_time': sim_time,
        'seed': seed
    }


def generate_random_obstacles(n_obstacles: int, bounds: np.ndarray,
                              rng: np.random.Generator,
                              goal_pos: np.ndarray) -> list:
    """
    Generate random rectangular obstacles.

    Args:
        n_obstacles: Number of obstacles to generate
        bounds: State space bounds
        rng: Random number generator
        goal_pos: Goal position (to avoid placing obstacles on goal)

    Returns:
        List of obstacle polygons (Nx4 arrays)
    """
    obstacles = []
    min_size = 1.0
    max_size = 3.0

    for _ in range(n_obstacles):
        # Try to find valid obstacle placement
        for attempt in range(20):
            # Random size
            width = rng.uniform(min_size, max_size)
            height = rng.uniform(min_size, max_size)

            # Random position (with margin from boundaries)
            margin = 1.0
            x = rng.uniform(bounds[0, 0] + margin, bounds[0, 1] - margin - width)
            y = rng.uniform(bounds[1, 0] + margin, bounds[1, 1] - margin - height)

            # Check if too close to goal
            center = np.array([x + width/2, y + height/2])
            if np.linalg.norm(center - goal_pos) > 2.5:
                # Create rectangle vertices
                obstacle = np.array([
                    [x, y],
                    [x + width, y],
                    [x + width, y + height],
                    [x, y + height]
                ])
                obstacles.append(obstacle)
                break

    return obstacles


def generate_random_goal(bounds: np.ndarray, rng: np.random.Generator) -> tuple:
    """
    Generate random goal position and region.

    Returns:
        goal_pos: Goal center position
        goal_region: Goal region polygon
    """
    margin = 2.0
    goal_size = 1.5

    x = rng.uniform(bounds[0, 0] + margin, bounds[0, 1] - margin)
    y = rng.uniform(bounds[1, 0] + margin, bounds[1, 1] - margin)

    goal_pos = np.array([x, y])

    goal_region = np.array([
        [x - goal_size/2, y - goal_size/2],
        [x + goal_size/2, y - goal_size/2],
        [x + goal_size/2, y + goal_size/2],
        [x - goal_size/2, y + goal_size/2]
    ])

    return goal_pos, goal_region


def generate_random_start(bounds: np.ndarray, obstacles: list,
                          goal_pos: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Generate random valid starting position.

    Ensures start is not inside obstacles and not too close to goal.
    """
    from shapely.geometry import Point, Polygon as ShapelyPolygon

    obstacle_polys = [ShapelyPolygon(obs) for obs in obstacles]

    for _ in range(100):
        margin = 1.0
        x = rng.uniform(bounds[0, 0] + margin, bounds[0, 1] - margin)
        y = rng.uniform(bounds[1, 0] + margin, bounds[1, 1] - margin)

        point = Point(x, y)

        # Check not in obstacles
        in_obstacle = any(poly.contains(point) for poly in obstacle_polys)

        # Check not too close to goal
        dist_to_goal = np.linalg.norm(np.array([x, y]) - goal_pos)

        if not in_obstacle and dist_to_goal > 3.0:
            return np.array([x, y])

    # Fallback: corner position
    return np.array([bounds[0, 0] + 1.0, bounds[1, 0] + 1.0])


def run_simulation(model, controller, x0, target, T, disturbance_mode='random'):
    """Run a single simulation."""
    controller.set_target(target)
    controller.reset()

    sim = Simulator(model, controller, disturbance_mode=disturbance_mode, seed=None)
    result = sim.run(x0, T)

    return result


def visualize_results(model, config, goal_pos, goal_region, obstacles,
                      all_results, abstraction):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(16, 10))

    # Main trajectory plot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_xlim(model.x_bounds[0])
    ax1.set_ylim(model.x_bounds[1])
    ax1.set_aspect('equal')
    ax1.set_title('Trajectories (All Controllers)', fontsize=12)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')

    # Draw grid cells
    for cell_id in range(abstraction.n_cells):
        center = abstraction.cells[cell_id]
        rect = Rectangle(
            (center[0] - abstraction.cell_width/2,
             center[1] - abstraction.cell_height/2),
            abstraction.cell_width, abstraction.cell_height,
            fill=False, edgecolor='lightgray', linewidth=0.3
        )
        ax1.add_patch(rect)

    # Draw obstacles
    for obs in obstacles:
        poly = Polygon(obs, facecolor='gray', edgecolor='black', alpha=0.6)
        ax1.add_patch(poly)

    # Draw goal region
    goal_poly = Polygon(goal_region, facecolor='lightgreen',
                        edgecolor='darkgreen', alpha=0.5, linewidth=2)
    ax1.add_patch(goal_poly)
    ax1.plot(goal_pos[0], goal_pos[1], 'g*', markersize=15, label='Goal')

    # Plot trajectories
    colors = {'P-Control': 'blue', 'LQR': 'red', 'ReachAvoid': 'purple'}

    for controller_name, runs in all_results.items():
        color = colors.get(controller_name, 'black')
        for i, result in enumerate(runs):
            label = controller_name if i == 0 else None
            ax1.plot(result.states[:, 0], result.states[:, 1], '-',
                     color=color, alpha=0.5, linewidth=1, label=label)
            ax1.plot(result.states[0, 0], result.states[0, 1], 'o',
                     color=color, markersize=6)

    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Success rate bar chart
    ax2 = fig.add_subplot(2, 2, 2)

    success_rates = {}
    for controller_name, runs in all_results.items():
        successes = sum(1 for r in runs
                        if np.linalg.norm(r.states[-1] - goal_pos) < 2.0)
        success_rates[controller_name] = successes / len(runs) * 100

    bars = ax2.bar(success_rates.keys(), success_rates.values(),
                   color=[colors.get(k, 'gray') for k in success_rates.keys()])
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Controller Success Rates')
    ax2.set_ylim(0, 105)

    for bar, rate in zip(bars, success_rates.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{rate:.0f}%', ha='center', fontsize=10)

    # Lyapunov evolution
    ax3 = fig.add_subplot(2, 2, 3)

    for controller_name, runs in all_results.items():
        color = colors.get(controller_name, 'black')
        for result in runs:
            V = result.lyapunov_values
            V_positive = np.maximum(V, 1e-10)
            ax3.semilogy(result.time, V_positive, '-', color=color, alpha=0.3)

    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('V(x) [log scale]')
    ax3.set_title('Lyapunov Function Evolution')
    ax3.grid(True, alpha=0.3)

    # Statistics table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # Compute statistics
    stats_data = []
    for controller_name, runs in all_results.items():
        final_errors = [np.linalg.norm(r.states[-1] - goal_pos) for r in runs]
        energies = [r.compute_metrics()['input_energy'] for r in runs]

        stats_data.append([
            controller_name,
            f"{np.mean(final_errors):.2f}",
            f"{np.std(final_errors):.2f}",
            f"{np.mean(energies):.2f}",
            f"{success_rates[controller_name]:.0f}%"
        ])

    table = ax4.table(
        cellText=stats_data,
        colLabels=['Controller', 'Mean Error', 'Std Error', 'Mean Energy', 'Success'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Performance Statistics', fontsize=12, pad=20)

    plt.tight_layout()
    return fig


def main():
    """Main interactive demo."""
    # Get user configuration
    config = get_user_input()

    print("\n" + "=" * 60)
    print("RUNNING SIMULATION")
    print("=" * 60)

    # Set up random generator
    if config['seed'] is not None:
        rng = np.random.default_rng(config['seed'])
        print(f"\nUsing seed: {config['seed']}")
    else:
        rng = np.random.default_rng()
        print("\nUsing random seed")

    # Create model
    model = IntegratorModel(tau=0.1)
    print(f"\nModel: {model}")

    # Generate random goal
    goal_pos, goal_region = generate_random_goal(model.x_bounds, rng)
    print(f"Goal position: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")

    # Generate random obstacles
    obstacles = generate_random_obstacles(
        config['n_obstacles'], model.x_bounds, rng, goal_pos)
    print(f"Generated {len(obstacles)} obstacles")

    # Create grid abstraction
    print(f"\nCreating grid abstraction (resolution={config['resolution']})...")
    abstraction = GridAbstraction(
        model.x_bounds,
        resolution=(config['resolution'], config['resolution']),
        model=model
    )
    abstraction.set_goal_region(goal_region)
    abstraction.set_obstacles(obstacles)

    print(f"  Total cells: {abstraction.n_cells}")
    print(f"  Safe cells: {len(abstraction.safe_cells)}")
    print(f"  Obstacle cells: {len(abstraction.obstacle_cells)}")
    print(f"  Goal cells: {len(abstraction.goal_cells)}")

    # Create planner
    print("\nComputing reach-avoid plan...")
    planner = ReachAvoidPlanner(abstraction)
    planner.compute_value_function()
    planner.compute_policy()

    # Set up controllers
    controllers = {
        'P-Control': ProportionalController(kp=0.6, name="P-Control"),
        'LQR': LQRController(name="LQR"),
        'ReachAvoid': ReachAvoidController(planner, waypoint_tolerance=0.6,
                                            kp=0.8, name="ReachAvoid")
    }

    # Design LQR
    controllers['LQR'].design_for_model(model)

    # Set model for all controllers
    for ctrl in controllers.values():
        ctrl.set_model(model)

    # Run simulations
    print(f"\nRunning {config['n_runs']} simulations per controller...")
    all_results = {name: [] for name in controllers.keys()}

    for run_idx in range(config['n_runs']):
        # Generate random start position
        x0 = generate_random_start(model.x_bounds, obstacles, goal_pos, rng)
        print(f"  Run {run_idx + 1}: Start=({x0[0]:.2f}, {x0[1]:.2f})")

        for name, controller in controllers.items():
            result = run_simulation(
                model, controller, x0, goal_pos,
                config['sim_time'], 'random'
            )
            all_results[name].append(result)

            final_dist = np.linalg.norm(result.states[-1] - goal_pos)
            status = "OK" if final_dist < 2.0 else "MISS"
            print(f"    {name}: Final dist={final_dist:.2f} [{status}]")

    # Visualize results
    print("\nGenerating visualization...")
    fig = visualize_results(model, config, goal_pos, goal_region,
                            obstacles, all_results, abstraction)

    # Save figure
    os.makedirs('report/figures', exist_ok=True)
    output_path = 'report/figures/interactive_demo_results.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")

    # Show plot
    plt.show()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
