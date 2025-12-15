"""
Animation export functionality for robot simulations.

Supports exporting trajectories as:
- MP4 video files (requires ffmpeg)
- GIF animations (requires pillow/imageio)
- HTML5 animations (for Jupyter notebooks)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Polygon, Circle, FancyArrow
from typing import Optional, List, Dict, Tuple, Union
import os


class AnimationExporter:
    """
    Export robot trajectory animations to various formats.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 10), dpi: int = 100):
        """
        Initialize animation exporter.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Resolution for output files
        """
        self.figsize = figsize
        self.dpi = dpi

    def animate_integrator(self, result, model=None,
                           obstacles: List[np.ndarray] = None,
                           goal: np.ndarray = None,
                           title: str = "Integrator Robot",
                           fps: int = 20) -> FuncAnimation:
        """
        Create animation for 2D integrator robot.

        Args:
            result: SimulationResult with states and time
            model: IntegratorModel (optional, for bounds)
            obstacles: List of obstacle polygons
            goal: Goal region polygon
            title: Animation title
            fps: Frames per second

        Returns:
            FuncAnimation object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Setup bounds
        if model is not None:
            ax.set_xlim(model.x_bounds[0])
            ax.set_ylim(model.x_bounds[1])
        else:
            margin = 1.0
            ax.set_xlim([result.states[:, 0].min() - margin,
                         result.states[:, 0].max() + margin])
            ax.set_ylim([result.states[:, 1].min() - margin,
                         result.states[:, 1].max() + margin])

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)

        # Draw obstacles
        if obstacles:
            for obs in obstacles:
                poly = Polygon(obs, facecolor='gray', edgecolor='black',
                              alpha=0.7, zorder=1)
                ax.add_patch(poly)

        # Draw goal
        if goal is not None:
            goal_poly = Polygon(goal, facecolor='green', edgecolor='darkgreen',
                               alpha=0.3, zorder=1)
            ax.add_patch(goal_poly)

        # Trajectory trace
        trace, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5, zorder=2)

        # Robot marker
        robot = Circle((0, 0), 0.2, facecolor='blue', edgecolor='darkblue',
                       zorder=3)
        ax.add_patch(robot)

        # Time text
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top')

        # Frame skip for performance
        n_frames = len(result.time)
        skip = max(1, n_frames // (fps * 10))  # ~10 second animation

        def init():
            trace.set_data([], [])
            robot.center = (result.states[0, 0], result.states[0, 1])
            time_text.set_text('')
            return trace, robot, time_text

        def animate(i):
            frame = min(i * skip, n_frames - 1)
            x, y = result.states[frame, 0], result.states[frame, 1]

            trace.set_data(result.states[:frame+1, 0], result.states[:frame+1, 1])
            robot.center = (x, y)
            time_text.set_text(f't = {result.time[frame]:.2f}s')

            return trace, robot, time_text

        n_animation_frames = (n_frames - 1) // skip + 1
        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=n_animation_frames,
                            interval=1000/fps, blit=True)

        return anim

    def animate_unicycle(self, result, model=None,
                         target: np.ndarray = None,
                         title: str = "Unicycle Robot",
                         fps: int = 20) -> FuncAnimation:
        """
        Create animation for unicycle (differential drive) robot.

        Args:
            result: SimulationResult with states [x, y, theta]
            model: UnicycleModel (optional)
            target: Target state [x, y, theta]
            title: Animation title
            fps: Frames per second

        Returns:
            FuncAnimation object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Setup bounds
        margin = 2.0
        ax.set_xlim([result.states[:, 0].min() - margin,
                     result.states[:, 0].max() + margin])
        ax.set_ylim([result.states[:, 1].min() - margin,
                     result.states[:, 1].max() + margin])

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)

        # Target marker
        if target is not None:
            ax.plot(target[0], target[1], 'r*', markersize=15, zorder=1)
            # Target heading arrow
            dx = 0.5 * np.cos(target[2])
            dy = 0.5 * np.sin(target[2])
            ax.arrow(target[0], target[1], dx, dy,
                     head_width=0.15, head_length=0.1,
                     fc='red', ec='red', alpha=0.5, zorder=1)

        # Trajectory trace
        trace, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5, zorder=2)

        # Robot body (triangle pointing in heading direction)
        robot_size = 0.3
        robot, = ax.plot([], [], 'b-', linewidth=2, zorder=3)
        robot_fill = Polygon([(0, 0)], facecolor='blue', edgecolor='darkblue',
                            alpha=0.7, zorder=3)
        ax.add_patch(robot_fill)

        # Time text
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            fontsize=12, verticalalignment='top')

        # Frame skip
        n_frames = len(result.time)
        skip = max(1, n_frames // (fps * 10))

        def get_robot_triangle(x, y, theta, size=0.3):
            """Get triangle vertices for robot visualization."""
            # Triangle pointing in heading direction
            front = np.array([x + size * np.cos(theta),
                              y + size * np.sin(theta)])
            back_left = np.array([x + size * 0.5 * np.cos(theta + 2.5),
                                   y + size * 0.5 * np.sin(theta + 2.5)])
            back_right = np.array([x + size * 0.5 * np.cos(theta - 2.5),
                                    y + size * 0.5 * np.sin(theta - 2.5)])
            return np.array([front, back_left, back_right])

        def init():
            trace.set_data([], [])
            verts = get_robot_triangle(result.states[0, 0],
                                        result.states[0, 1],
                                        result.states[0, 2])
            robot_fill.set_xy(verts)
            time_text.set_text('')
            return trace, robot_fill, time_text

        def animate(i):
            frame = min(i * skip, n_frames - 1)
            x, y, theta = result.states[frame, :3]

            trace.set_data(result.states[:frame+1, 0], result.states[:frame+1, 1])
            verts = get_robot_triangle(x, y, theta)
            robot_fill.set_xy(verts)
            time_text.set_text(f't = {result.time[frame]:.2f}s')

            return trace, robot_fill, time_text

        n_animation_frames = (n_frames - 1) // skip + 1
        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=n_animation_frames,
                            interval=1000/fps, blit=True)

        return anim

    def animate_manipulator(self, result, model,
                            target: np.ndarray = None,
                            title: str = "Two-Link Manipulator",
                            fps: int = 30) -> FuncAnimation:
        """
        Create animation for two-link manipulator.

        Args:
            result: SimulationResult with states [theta1, theta2, dtheta1, dtheta2]
            model: TwoLinkManipulator
            target: Target state [theta1, theta2, ...]
            title: Animation title
            fps: Frames per second

        Returns:
            FuncAnimation object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Setup bounds based on arm reach
        reach = model.l1 + model.l2 + 0.2
        ax.set_xlim([-reach, reach])
        ax.set_ylim([-reach, reach])

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(title)

        # Draw base
        base = Circle((0, 0), 0.05, facecolor='black', zorder=4)
        ax.add_patch(base)

        # Draw target configuration (ghost)
        if target is not None:
            p1_t, p2_t = model.forward_kinematics(target[:2])
            ax.plot([0, p1_t[0], p2_t[0]], [0, p1_t[1], p2_t[1]],
                    'o--', color='gray', linewidth=2, markersize=6,
                    alpha=0.4, label='Target', zorder=1)

        # Arm links
        link1, = ax.plot([], [], 'o-', color='blue', linewidth=4,
                         markersize=8, zorder=3)
        link2, = ax.plot([], [], 'o-', color='red', linewidth=4,
                         markersize=8, zorder=3)

        # End-effector trace
        ee_trace, = ax.plot([], [], '-', color='lightblue',
                            linewidth=1, alpha=0.5, zorder=2)
        ee_history = []

        # Time and angle text
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                            fontsize=11, verticalalignment='top')
        angle_text = ax.text(0.02, 0.88, '', transform=ax.transAxes,
                             fontsize=10, verticalalignment='top')

        # Frame skip
        n_frames = len(result.time)
        skip = max(1, n_frames // (fps * 10))

        def init():
            link1.set_data([], [])
            link2.set_data([], [])
            ee_trace.set_data([], [])
            ee_history.clear()
            time_text.set_text('')
            angle_text.set_text('')
            return link1, link2, ee_trace, time_text, angle_text

        def animate(i):
            frame = min(i * skip, n_frames - 1)
            theta = result.states[frame, :2]
            t = result.time[frame]

            p1, p2 = model.forward_kinematics(theta)

            link1.set_data([0, p1[0]], [0, p1[1]])
            link2.set_data([p1[0], p2[0]], [p1[1], p2[1]])

            ee_history.append(p2.copy())
            if len(ee_history) > 1:
                ee_arr = np.array(ee_history)
                ee_trace.set_data(ee_arr[:, 0], ee_arr[:, 1])

            time_text.set_text(f't = {t:.2f}s')
            angle_text.set_text(f'theta1={np.degrees(theta[0]):.1f} deg, '
                                f'theta2={np.degrees(theta[1]):.1f} deg')

            return link1, link2, ee_trace, time_text, angle_text

        n_animation_frames = (n_frames - 1) // skip + 1
        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=n_animation_frames,
                            interval=1000/fps, blit=True)

        return anim

    def save_mp4(self, anim: FuncAnimation, filename: str, fps: int = 20):
        """
        Save animation as MP4 video.

        Requires ffmpeg to be installed and in PATH.

        Args:
            anim: FuncAnimation object
            filename: Output filename (should end in .mp4)
            fps: Frames per second
        """
        if not filename.endswith('.mp4'):
            filename += '.mp4'

        try:
            writer = FFMpegWriter(fps=fps, metadata={'title': 'Robot Animation'},
                                  bitrate=1800)
            anim.save(filename, writer=writer, dpi=self.dpi)
            print(f"Animation saved to {filename}")
        except Exception as e:
            print(f"Error saving MP4 (ffmpeg may not be installed): {e}")
            print("Trying to save as GIF instead...")
            self.save_gif(anim, filename.replace('.mp4', '.gif'), fps=fps)

    def save_gif(self, anim: FuncAnimation, filename: str, fps: int = 20):
        """
        Save animation as GIF.

        Args:
            anim: FuncAnimation object
            filename: Output filename (should end in .gif)
            fps: Frames per second
        """
        if not filename.endswith('.gif'):
            filename += '.gif'

        try:
            writer = PillowWriter(fps=fps)
            anim.save(filename, writer=writer, dpi=self.dpi)
            print(f"Animation saved to {filename}")
        except Exception as e:
            print(f"Error saving GIF: {e}")

    def save_html(self, anim: FuncAnimation, filename: str):
        """
        Save animation as HTML5 (for viewing in browser).

        Args:
            anim: FuncAnimation object
            filename: Output filename (should end in .html)
        """
        if not filename.endswith('.html'):
            filename += '.html'

        try:
            html = anim.to_jshtml()
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Animation saved to {filename}")
        except Exception as e:
            print(f"Error saving HTML: {e}")


def export_simulation(result, model, output_path: str,
                      format: str = 'gif',
                      robot_type: str = 'auto',
                      **kwargs):
    """
    Convenience function to export simulation result as animation.

    Args:
        result: SimulationResult object
        model: Robot model
        output_path: Output file path (extension added automatically)
        format: 'gif', 'mp4', or 'html'
        robot_type: 'integrator', 'unicycle', 'manipulator', or 'auto'
        **kwargs: Additional arguments for animation (target, obstacles, etc.)
    """
    exporter = AnimationExporter()

    # Auto-detect robot type
    if robot_type == 'auto':
        n_states = result.states.shape[1]
        if n_states == 2:
            robot_type = 'integrator'
        elif n_states == 3:
            robot_type = 'unicycle'
        elif n_states == 4:
            robot_type = 'manipulator'
        else:
            robot_type = 'integrator'  # Default

    # Create animation
    if robot_type == 'integrator':
        anim = exporter.animate_integrator(result, model, **kwargs)
    elif robot_type == 'unicycle':
        anim = exporter.animate_unicycle(result, model, **kwargs)
    elif robot_type == 'manipulator':
        anim = exporter.animate_manipulator(result, model, **kwargs)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

    # Save
    if format.lower() == 'mp4':
        exporter.save_mp4(anim, output_path)
    elif format.lower() == 'gif':
        exporter.save_gif(anim, output_path)
    elif format.lower() == 'html':
        exporter.save_html(anim, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")

    plt.close()
    return output_path


def create_comparison_animation(results: List, labels: List[str],
                                model, robot_type: str = 'integrator',
                                figsize: Tuple[int, int] = (12, 8),
                                fps: int = 20) -> FuncAnimation:
    """
    Create side-by-side comparison animation of multiple controllers.

    Args:
        results: List of SimulationResult objects
        labels: List of controller names
        model: Robot model
        robot_type: Type of robot
        figsize: Figure size
        fps: Frames per second

    Returns:
        FuncAnimation object
    """
    n_controllers = len(results)
    fig, axes = plt.subplots(1, n_controllers, figsize=figsize)
    if n_controllers == 1:
        axes = [axes]

    # Setup each subplot
    robots = []
    traces = []
    time_texts = []

    for i, (result, label, ax) in enumerate(zip(results, labels, axes)):
        margin = 1.0
        if robot_type == 'integrator':
            ax.set_xlim([result.states[:, 0].min() - margin,
                         result.states[:, 0].max() + margin])
            ax.set_ylim([result.states[:, 1].min() - margin,
                         result.states[:, 1].max() + margin])
        elif robot_type == 'unicycle':
            ax.set_xlim([result.states[:, 0].min() - margin,
                         result.states[:, 0].max() + margin])
            ax.set_ylim([result.states[:, 1].min() - margin,
                         result.states[:, 1].max() + margin])

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(label)

        trace, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5)
        traces.append(trace)

        if robot_type == 'integrator':
            robot = Circle((0, 0), 0.2, facecolor='blue', edgecolor='darkblue')
            ax.add_patch(robot)
            robots.append(robot)
        else:
            robot, = ax.plot([], [], 'b-', linewidth=2)
            robots.append(robot)

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10)
        time_texts.append(time_text)

    # Find max frames
    max_frames = max(len(r.time) for r in results)
    skip = max(1, max_frames // (fps * 10))

    def init():
        elements = []
        for i, result in enumerate(results):
            traces[i].set_data([], [])
            if robot_type == 'integrator':
                robots[i].center = (result.states[0, 0], result.states[0, 1])
            time_texts[i].set_text('')
            elements.extend([traces[i], robots[i], time_texts[i]])
        return elements

    def animate(frame_idx):
        elements = []
        for i, result in enumerate(results):
            f = min(frame_idx * skip, len(result.time) - 1)

            traces[i].set_data(result.states[:f+1, 0], result.states[:f+1, 1])

            if robot_type == 'integrator':
                robots[i].center = (result.states[f, 0], result.states[f, 1])

            time_texts[i].set_text(f't = {result.time[f]:.2f}s')
            elements.extend([traces[i], robots[i], time_texts[i]])

        return elements

    n_animation_frames = (max_frames - 1) // skip + 1
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=n_animation_frames,
                        interval=1000/fps, blit=True)

    plt.tight_layout()
    return anim
