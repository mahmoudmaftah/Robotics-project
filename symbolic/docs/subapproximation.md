# Reachability Analysis via Jacobian-based Growth Bounds

Based on **Section 1.1.3** of the document, the "Jacobian method" (also known as the **growth bound method**) is an alternative to the monotone method for over-approximating the reachable set during the symbolic abstraction phase. 

It is particularly useful because it applies to any system where the function $f$ is differentiable and its derivatives are bounded, even if the system is not monotone.

### 1. Requirements and Assumptions
For the system $x(t+1) = f(x, u, w)$, the method requires that:
*   The function $f$ is **differentiable**.
*   The partial derivatives (Jacobians) are **uniformly bounded**. There must exist constant matrices $D_X \in \mathbb{R}^{n_x \times n_x}$ and $D_W \in \mathbb{R}^{n_x \times n_w}$ such that:
    $$\left| \frac{\partial f}{\partial x}(x, u, w) \right| \leq D_X \quad \text{and} \quad \left| \frac{\partial f}{\partial w}(x, u, w) \right| \leq D_W$$
    *(Inequalities are interpreted coefficient by coefficient).*

### 2. Discretization Parameters
For an interval of initial symbolic states $cl(X_\xi) = [\underline{x}, \bar{x}]$, we define:
*   **The center:** $x^* = \frac{\bar{x} + \underline{x}}{2}$
*   **The radius (half-width):** $\delta_x = \frac{\bar{x} - \underline{x}}{2}$

Similarly, for the perturbation interval $W = [\underline{w}, \bar{w}]$, we define the center $w^*$ and the half-width $\delta_w$.

### 3. The Over-approximation Formula
To calculate the ensemble of potential successor states, the method computes the image of the center and expands it by a safety margin derived from the growth bounds:

The set of reachable states is over-approximated by the interval:
$$[f(x^*, u_\sigma, w^*) - \Delta, \quad f(x^*, u_\sigma, w^*) + \Delta]$$

Where the expansion term **$\Delta$** is:
$$\Delta = D_X \delta_x + D_W \delta_w$$

### 4. Intuition and Visualization (Figure 3b)
*   **Computational Efficiency:** Instead of calculating the image of every point in the interval (which is impossible) or just the corners (which is insufficient), we only calculate the image of the **center point** $f(x^*, u_\sigma, w^*)$.
*   **Expansion:** We then "grow" a box around this new center. The size of this box ($\Delta$) is proportional to the maximum possible variation of the system, ensured by the bounds on the Jacobian.
*   **Result:** This red box (as shown in Figure 3b) is guaranteed to contain the entire "blue" set of actual possible future states, ensuring the symbolic model is a valid abstraction.