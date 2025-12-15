#!/usr/bin/env python3
"""
Interactive Canvas for Drawing Obstacles and Goals.

Allows users to:
1. Click to draw polygonal obstacles
2. Click to draw goal region
3. Click to set start position
4. Run simulation with chosen controller

Uses matplotlib event handling for interactive drawing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.widgets import Button, RadioButtons
from typing import List, Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.integrator import IntegratorModel
from controllers import ProportionalController, LQRController
from controllers.mpc import MPCController
from symbolic.grid_abstraction import GridAbstraction
from symbolic.reach_avoid import ReachAvoidPlanner, ReachAvoidController
from sim.simulator import Simulator
from sim.animation import export_simulation


class InteractiveCanvas:
    """
    Interactive matplotlib canvas for drawing robot control scenarios.

    Modes:
        - 'obstacle': Draw polygonal obstacles (right-click to finish)
        - 'goal': Draw goal region polygon (right-click to finish)
        - 'start': Click to set start position
        - 'run': Execute simulation
    """

    def __init__(self, bounds: np.ndarray = None):
        """
        Initialize interactive canvas.

        Args:
            bounds: Workspace bounds [[x_min, x_max], [y_min, y_max]]
        """
        if bounds is None:
            bounds = np.array([[-10, 10], [-10, 10]])
        self.bounds = bounds

        # Drawing state
        self.mode = 'obstacle'  # 'obstacle', 'goal', 'start', 'run'
        self.current_polygon: List[Tuple[float, float]] = []
        self.obstacles: List[np.ndarray] = []
        self.goal_region: Optional[np.ndarray] = None
        self.start_pos: Optional[np.ndarray] = None

        # Controller selection
        self.controller_type = 'LQR'

        # Simulation results
        self.result = None

        # Setup figure
        self._setup_figure()

    def _setup_figure(self):
        """Setup matplotlib figure with buttons and canvas."""
        self.fig = plt.figure(figsize=(14, 10))

        # Main canvas axes
        self.ax = self.fig.add_axes([0.1, 0.25, 0.6, 0.7])
        self.ax.set_xlim(self.bounds[0])
        self.ax.set_ylim(self.bounds[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self._update_title()

        # Instructions text
        self.info_ax = self.fig.add_axes([0.72, 0.5, 0.26, 0.45])
        self.info_ax.axis('off')
        instructions = """INSTRUCTIONS:

1. OBSTACLE MODE (default)
   - Left-click to add vertices
   - Right-click to finish polygon
   - Draw multiple obstacles

2. GOAL MODE
   - Left-click to add vertices
   - Right-click to finish polygon

3. START MODE
   - Left-click to place start

4. RUN SIMULATION
   - Click 'Run' button
   - Trajectory will be displayed

KEYBOARD SHORTCUTS:
  'o' - Obstacle mode
  'g' - Goal mode
  's' - Start mode
  'c' - Clear all
  'r' - Run simulation
  'e' - Export animation"""
        self.info_ax.text(0, 1, instructions, transform=self.info_ax.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace')

        # Mode buttons
        btn_width = 0.08
        btn_height = 0.05
        btn_y = 0.12

        self.btn_obstacle = Button(
            plt.axes([0.1, btn_y, btn_width, btn_height]), 'Obstacle')
        self.btn_obstacle.on_clicked(lambda e: self._set_mode('obstacle'))

        self.btn_goal = Button(
            plt.axes([0.19, btn_y, btn_width, btn_height]), 'Goal')
        self.btn_goal.on_clicked(lambda e: self._set_mode('goal'))

        self.btn_start = Button(
            plt.axes([0.28, btn_y, btn_width, btn_height]), 'Start')
        self.btn_start.on_clicked(lambda e: self._set_mode('start'))

        self.btn_clear = Button(
            plt.axes([0.37, btn_y, btn_width, btn_height]), 'Clear')
        self.btn_clear.on_clicked(lambda e: self._clear_all())

        self.btn_run = Button(
            plt.axes([0.46, btn_y, btn_width, btn_height]), 'Run')
        self.btn_run.on_clicked(lambda e: self._run_simulation())

        self.btn_export = Button(
            plt.axes([0.55, btn_y, btn_width, btn_height]), 'Export')
        self.btn_export.on_clicked(lambda e: self._export_animation())

        # Controller selection
        self.radio_ax = plt.axes([0.72, 0.25, 0.15, 0.2])
        self.radio = RadioButtons(
            self.radio_ax,
            ('LQR', 'Proportional', 'MPC', 'Reach-Avoid'),
            active=0
        )
        self.radio.on_clicked(self._set_controller)
        self.radio_ax.set_title('Controller', fontsize=10)

        # Status bar
        self.status_ax = self.fig.add_axes([0.1, 0.02, 0.8, 0.05])
        self.status_ax.axis('off')
        self.status_text = self.status_ax.text(
            0.5, 0.5, 'Mode: OBSTACLE | Click to add vertices, right-click to finish',
            transform=self.status_ax.transAxes,
            ha='center', va='center', fontsize=10
        )

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Drawing elements
        self.temp_line, = self.ax.plot([], [], 'k--', linewidth=1, alpha=0.5)
        self.temp_points, = self.ax.plot([], [], 'ko', markersize=5)

    def _update_title(self):
        """Update plot title with current mode."""
        mode_colors = {
            'obstacle': 'red',
            'goal': 'green',
            'start': 'blue',
            'run': 'black'
        }
        self.ax.set_title(f'Interactive Robot Control Canvas - Mode: {self.mode.upper()}',
                         color=mode_colors.get(self.mode, 'black'))

    def _set_mode(self, mode: str):
        """Set current drawing mode."""
        # Finish any current polygon
        if self.current_polygon:
            self._finish_current_polygon()

        self.mode = mode
        self._update_title()
        self._update_status()
        self.fig.canvas.draw_idle()

    def _set_controller(self, label: str):
        """Set controller type."""
        self.controller_type = label
        self._update_status()

    def _update_status(self):
        """Update status bar text."""
        status_msgs = {
            'obstacle': 'Mode: OBSTACLE | Click to add vertices, right-click to finish',
            'goal': 'Mode: GOAL | Click to add vertices, right-click to finish',
            'start': 'Mode: START | Click to place start position',
            'run': f'Ready to run with {self.controller_type} controller'
        }
        self.status_text.set_text(status_msgs.get(self.mode, ''))

    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:  # Left click
            if self.mode == 'obstacle':
                self._add_polygon_point(x, y)
            elif self.mode == 'goal':
                self._add_polygon_point(x, y)
            elif self.mode == 'start':
                self._set_start(x, y)

        elif event.button == 3:  # Right click
            if self.mode in ('obstacle', 'goal'):
                self._finish_current_polygon()

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'o':
            self._set_mode('obstacle')
        elif event.key == 'g':
            self._set_mode('goal')
        elif event.key == 's':
            self._set_mode('start')
        elif event.key == 'c':
            self._clear_all()
        elif event.key == 'r':
            self._run_simulation()
        elif event.key == 'e':
            self._export_animation()

    def _add_polygon_point(self, x: float, y: float):
        """Add a point to the current polygon."""
        self.current_polygon.append((x, y))

        # Update temporary visualization
        if len(self.current_polygon) > 0:
            pts = np.array(self.current_polygon)
            self.temp_points.set_data(pts[:, 0], pts[:, 1])

            if len(self.current_polygon) > 1:
                # Show lines connecting points
                closed_pts = np.vstack([pts, pts[0]])
                self.temp_line.set_data(closed_pts[:, 0], closed_pts[:, 1])

    def _finish_current_polygon(self):
        """Finish the current polygon and add it to the list."""
        if len(self.current_polygon) < 3:
            self.current_polygon = []
            self.temp_points.set_data([], [])
            self.temp_line.set_data([], [])
            return

        polygon = np.array(self.current_polygon)

        if self.mode == 'obstacle':
            self.obstacles.append(polygon)
            # Draw obstacle
            patch = Polygon(polygon, facecolor='gray', edgecolor='black',
                           alpha=0.7)
            self.ax.add_patch(patch)

        elif self.mode == 'goal':
            self.goal_region = polygon
            # Remove old goal if exists
            for patch in self.ax.patches:
                if hasattr(patch, '_is_goal') and patch._is_goal:
                    patch.remove()
            # Draw goal
            patch = Polygon(polygon, facecolor='green', edgecolor='darkgreen',
                           alpha=0.4)
            patch._is_goal = True
            self.ax.add_patch(patch)

        # Clear temporary drawing
        self.current_polygon = []
        self.temp_points.set_data([], [])
        self.temp_line.set_data([], [])

    def _set_start(self, x: float, y: float):
        """Set the start position."""
        self.start_pos = np.array([x, y])

        # Remove old start marker if exists
        for artist in self.ax.artists:
            if hasattr(artist, '_is_start') and artist._is_start:
                artist.remove()

        # Draw start marker
        circle = Circle((x, y), 0.3, facecolor='blue', edgecolor='darkblue',
                        zorder=10)
        circle._is_start = True
        self.ax.add_artist(circle)

        # Add 'S' label
        self.ax.annotate('S', (x, y), ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white', zorder=11)

    def _clear_all(self):
        """Clear all drawings."""
        self.obstacles = []
        self.goal_region = None
        self.start_pos = None
        self.current_polygon = []
        self.result = None

        # Clear axes and redraw
        self.ax.clear()
        self.ax.set_xlim(self.bounds[0])
        self.ax.set_ylim(self.bounds[1])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self._update_title()

        # Recreate temp drawing elements
        self.temp_line, = self.ax.plot([], [], 'k--', linewidth=1, alpha=0.5)
        self.temp_points, = self.ax.plot([], [], 'ko', markersize=5)

        self._update_status()
        self.fig.canvas.draw_idle()
        print("Canvas cleared.")

    def _run_simulation(self):
        """Run simulation with current configuration."""
        # Validate configuration
        if self.start_pos is None:
            print("Error: Please set a start position first!")
            self.status_text.set_text("ERROR: Set start position first (click 'Start' then click on canvas)")
            self.fig.canvas.draw_idle()
            return

        if self.goal_region is None and self.controller_type == 'Reach-Avoid':
            print("Error: Reach-Avoid requires a goal region!")
            self.status_text.set_text("ERROR: Draw a goal region first for Reach-Avoid")
            self.fig.canvas.draw_idle()
            return

        print(f"\nRunning simulation with {self.controller_type} controller...")
        print(f"  Start: {self.start_pos}")
        print(f"  Obstacles: {len(self.obstacles)}")
        print(f"  Goal: {'Set' if self.goal_region is not None else 'Not set'}")

        self.status_text.set_text(f"Running {self.controller_type} simulation...")
        self.fig.canvas.draw_idle()
        plt.pause(0.1)

        # Create model
        model = IntegratorModel(tau=0.1)

        # Set target (center of goal region or origin)
        if self.goal_region is not None:
            target = np.mean(self.goal_region, axis=0)
        else:
            target = np.zeros(2)

        # Create controller based on selection
        if self.controller_type == 'LQR':
            controller = LQRController()
            controller.design_for_model(model)
            controller.set_target(target)

        elif self.controller_type == 'Proportional':
            controller = ProportionalController(kp=0.5)
            controller.set_model(model)
            controller.set_target(target)

        elif self.controller_type == 'MPC':
            controller = MPCController(horizon=10, Q=np.eye(2), R=0.1*np.eye(2))
            controller.set_model(model)
            controller.set_target(target)

        elif self.controller_type == 'Reach-Avoid':
            # Create grid abstraction
            abstraction = GridAbstraction(
                bounds=self.bounds,
                resolution=(25, 25),
                model=model
            )
            abstraction.set_goal_region(self.goal_region)
            for obs in self.obstacles:
                abstraction.add_obstacle(obs)

            # Plan
            planner = ReachAvoidPlanner(abstraction)
            planner.compute_value_function()
            planner.compute_policy()

            controller = ReachAvoidController(planner, waypoint_tolerance=0.5, kp=0.8)
            controller.set_model(model)

        # Run simulation
        sim = Simulator(model, controller, disturbance_mode='random', seed=42)
        self.result = sim.run(self.start_pos, T=100.0, target=target)

        # Plot trajectory
        self._plot_trajectory()

        # Compute and display metrics
        metrics = self.result.compute_metrics()
        print(f"\nSimulation Results:")
        print(f"  Final error: {self.result.error_norms[-1]:.4f}")
        print(f"  Settling time: {metrics.get('settling_time', 'N/A')}")
        print(f"  Input energy: {metrics.get('input_energy', 0):.4f}")

        reached = self.result.error_norms[-1] < 0.5
        status = "SUCCESS" if reached else "IN PROGRESS"
        self.status_text.set_text(
            f"{status} | Final error: {self.result.error_norms[-1]:.3f} | "
            f"Energy: {metrics.get('input_energy', 0):.2f}"
        )
        self.fig.canvas.draw_idle()

    def _plot_trajectory(self):
        """Plot the simulation trajectory on the canvas."""
        if self.result is None:
            return

        # Plot trajectory
        self.ax.plot(self.result.states[:, 0], self.result.states[:, 1],
                    'b-', linewidth=2, alpha=0.7, label='Trajectory')

        # Mark end position
        self.ax.plot(self.result.states[-1, 0], self.result.states[-1, 1],
                    'ro', markersize=10, label='End')

        self.ax.legend(loc='upper right')

    def _export_animation(self):
        """Export animation of the simulation."""
        if self.result is None:
            print("Error: Run a simulation first!")
            self.status_text.set_text("ERROR: Run simulation before exporting")
            self.fig.canvas.draw_idle()
            return

        print("Exporting animation...")
        self.status_text.set_text("Exporting animation to 'interactive_result.gif'...")
        self.fig.canvas.draw_idle()
        plt.pause(0.1)

        try:
            from models.integrator import IntegratorModel
            model = IntegratorModel(tau=0.1)

            export_simulation(
                self.result, model,
                'interactive_result',
                format='gif',
                robot_type='integrator',
                obstacles=self.obstacles,
                goal=self.goal_region,
                title=f'{self.controller_type} Control'
            )

            print("Animation saved to 'interactive_result.gif'")
            self.status_text.set_text("Animation saved to 'interactive_result.gif'")
        except Exception as e:
            print(f"Export failed: {e}")
            self.status_text.set_text(f"Export failed: {e}")

        self.fig.canvas.draw_idle()

    def show(self):
        """Display the interactive canvas."""
        print("\n" + "="*60)
        print("INTERACTIVE ROBOT CONTROL CANVAS")
        print("="*60)
        print("\nInstructions:")
        print("  1. Draw obstacles: Click points, right-click to finish")
        print("  2. Draw goal: Switch to Goal mode, draw polygon")
        print("  3. Set start: Switch to Start mode, click position")
        print("  4. Select controller and click Run")
        print("\nKeyboard shortcuts: o=obstacle, g=goal, s=start, c=clear, r=run")
        print("="*60 + "\n")

        plt.show()


def main():
    """Main entry point for interactive canvas."""
    canvas = InteractiveCanvas(bounds=np.array([[-10, 10], [-10, 10]]))
    canvas.show()


if __name__ == '__main__':
    main()
