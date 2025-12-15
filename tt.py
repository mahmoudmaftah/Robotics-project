import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class UnicycleRobot:
    def __init__(self, start_state, tau):
        # State: [x1, x2, x3] -> [x, y, theta]
        self.state = np.array(start_state, dtype=float)
        self.tau = tau
        self.path_x = [self.state[0]]
        self.path_y = [self.state[1]]
        
        # Constraints definition
        self.X_lim = [0, 10]
        self.Y_lim = [0, 10]
        self.U1_lim = [0.25, 1.0] # Linear velocity (cannot stop!)
        self.U2_lim = [-1.0, 1.0] # Angular velocity
        self.W_lim =  [-0.05, 0.05] # Disturbance bounds

    def normalize_angle(self, angle):
        """ Wraps angle to [-pi, pi] """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def step(self, u_command):
        """
        Apply inputs u_command: [u1, u2]
        """
        x1, x2, x3 = self.state

        # 1. Enforce Input Constraints (Clamping)
        u1 = np.clip(u_command[0], self.U1_lim[0], self.U1_lim[1])
        u2 = np.clip(u_command[1], self.U2_lim[0], self.U2_lim[1])

        # 2. Generate Random Disturbances (w)
        # Uniform distribution between -0.05 and 0.05
        w = np.random.uniform(self.W_lim[0], self.W_lim[1], size=3)

        # 3. System Dynamics (The Equations)
        # x1(t+1) = x1(t) + tau * (u1 * cos(x3) + w1)
        x1_next = x1 + self.tau * (u1 * np.cos(x3) + w[0])
        
        # x2(t+1) = x2(t) + tau * (u1 * sin(x3) + w2)
        x2_next = x2 + self.tau * (u1 * np.sin(x3) + w[1])
        
        # x3(t+1) = x3(t) + tau * (u2 + w3)
        x3_next = x3 + self.tau * (u2 + w[2])
        
        # Normalize the angle (mod 2pi)
        x3_next = self.normalize_angle(x3_next)

        # Update internal state
        self.state = np.array([x1_next, x2_next, x3_next])
        
        # Store for plotting
        self.path_x.append(x1_next)
        self.path_y.append(x2_next)

    def check_constraints(self):
        """ Returns True if robot is safe (inside the box), False if crashed """
        x, y, _ = self.state
        if (x < self.X_lim[0] or x > self.X_lim[1] or 
            y < self.Y_lim[0] or y > self.Y_lim[1]):
            return False
        return True

def run_simulation():
    # Simulation Parameters
    TAU = 0.1          # Sampling period
    STEPS = 200        # How long to run
    
    # Initialize Robot at center of room, facing East
    robot = UnicycleRobot(start_state=[5.0, 5.0, 0.0], tau=TAU)
    
    # Define a target to drive towards (e.g., top right corner)
    target = np.array([8.0, 8.0])

    # Setup Plotting
    plt.ion() # Interactive mode on
    fig, ax = plt.subplots(figsize=(8, 8))
    
    print("Simulation Started... (Close window to stop)")

    for k in range(STEPS):
        # --- SIMPLE CONTROLLER (Logic to drive the robot) ---
        # 1. Calculate error to target
        dx = target[0] - robot.state[0]
        dy = target[1] - robot.state[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # 2. Calculate desired heading
        desired_theta = np.arctan2(dy, dx)
        
        # 3. Calculate heading error
        angle_error = robot.normalize_angle(desired_theta - robot.state[2])
        
        # 4. Control Inputs (Proportional Control)
        # u1: Speed (If far, go fast. If close, slow down to min speed)
        # Remember: u1 cannot be 0! It is constrained to [0.25, 1.0]
        cmd_velocity = 0.5 * dist 
        
        # u2: Steering (Turn towards target)
        cmd_turn = 2.0 * angle_error
        
        # Send to robot (The class handles the clamping/constraints)
        robot.step([cmd_velocity, cmd_turn])

        # --- VISUALIZATION ---
        ax.clear()
        
        # 1. Draw Constraints (The Room)
        rect = patches.Rectangle((0,0), 10, 10, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        
        # 2. Draw Target
        ax.plot(target[0], target[1], 'rx', markersize=10, label='Target')
        
        # 3. Draw Path
        ax.plot(robot.path_x, robot.path_y, 'b--', linewidth=1, label='Path')
        
        # 4. Draw Robot (Arrow)
        # x, y, dx, dy
        arrow_len = 0.5
        ax.arrow(robot.state[0], robot.state[1], 
                 arrow_len * np.cos(robot.state[2]), 
                 arrow_len * np.sin(robot.state[2]), 
                 head_width=0.3, color='g', zorder=5)
        
        # Formatting
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.set_title(f"Step: {k} | Pos: ({robot.state[0]:.2f}, {robot.state[1]:.2f})")
        ax.legend(loc='upper left')
        ax.grid(True)
        
        plt.draw()
        plt.pause(0.01) # Small pause to create animation effect
        
        # Check Collision
        if not robot.check_constraints():
            print(f"CRASH! Robot hit the wall at step {k}")
            break
        
        # Check Goal Reached (Approximate)
        if dist < 0.5:
            print("Target reached! Moving target...")
            # Pick a new random target inside the box
            target = np.random.uniform(1, 9, size=2)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()