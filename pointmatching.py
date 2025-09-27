import numpy as np
import matplotlib.pyplot as plt

def calculate_polar_features(ref_point, target_points):
    """
    Calculates the angle and distance from a reference point to a list of target points.
    """
    features = []
    for point in target_points:
        vector = point - ref_point
        distance = np.linalg.norm(vector)
        angle_rad = np.arctan2(vector[1], vector[0])
        angle_deg = np.degrees(angle_rad)
        features.append((angle_deg, distance))
    return features

# --- System 1: Generate Points ---
# Generate 5 points (A, B, C, D, E) on a 90-degree arc
radius = 10.0
# Evenly space 5 points from 90° to 180°
angles_deg_sys1 = np.linspace(90, 180, 5) 
angles_rad_sys1 = np.deg2rad(angles_deg_sys1)

all_points_sys1 = np.array([
    [radius * np.cos(angle), radius * np.sin(angle)] for angle in angles_rad_sys1
])

# A is the first point on the curve and the reference point
A = all_points_sys1[0]
# B, C, D, E are the other four points
points_BCDE = all_points_sys1[1:]

# --- System 2: Generate Symmetrical Points ---
# Create 5 symmetrical points for the second system
translation_vector = np.array([25, 0])
all_points_sys2 = all_points_sys1 * np.array([-1, 1]) + translation_vector

# A' is the first point on the second curve and the reference point
A_prime = all_points_sys2[0]
# B', C', D', E' are the other four points
points_BCDE_prime = all_points_sys2[1:]

# --- Perform Calculations ---
# Calculate the feature matrix for System 1 from reference A
f_A = calculate_polar_features(A, points_BCDE)

# Calculate the feature matrix for System 2 from reference A'
f_A_prime = calculate_polar_features(A_prime, points_BCDE_prime)

# --- Display the Matrices ---
print("--- Corrected Matrices (A is on the curve) ---")
print("\nMatrix f(A) for System 1:")
print("[(Angle, Distance)]")
for angle, dist in f_A:
    print(f"  ({angle:7.2f}°, {dist:5.2f})")

print("\nMatrix f(A') for System 2:")
print("[(Angle, Distance)]")
for angle, dist in f_A_prime:
    print(f"  ({angle:7.2f}°, {dist:5.2f})")

# --- Visualize the Systems ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Corrected Visualization (Reference Point on Curve)', fontsize=16)

# Plot System 1
ax1.set_title("System 1")
ax1.plot(all_points_sys1[:, 0], all_points_sys1[:, 1], 'r-', lw=1.5, label="Curve ABCDE")
ax1.scatter(A[0], A[1], color='black', s=100, zorder=5, label="A (Ref)")
ax1.scatter(points_BCDE[:, 0], points_BCDE[:, 1], color='red')
for point in points_BCDE:
    ax1.plot([A[0], point[0]], [A[1], point[1]], 'k--', lw=0.8)
ax1.axhline(0, color='gray', lw=0.5)
ax1.axvline(0, color='gray', lw=0.5)
ax1.set_xlim(-15, 15)
ax1.set_ylim(-2, 12)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, linestyle=':')
ax1.legend()

# Plot System 2
ax2.set_title("System 2")
ax2.plot(all_points_sys2[:, 0], all_points_sys2[:, 1], 'b-', lw=1.5, label="Curve A'B'C'D'E'")
ax2.scatter(A_prime[0], A_prime[1], color='black', s=100, zorder=5, label="A' (Ref)")
ax2.scatter(points_BCDE_prime[:, 0], points_BCDE_prime[:, 1], color='blue')
for point in points_BCDE_prime:
    ax2.plot([A_prime[0], point[0]], [A_prime[1], point[1]], 'k--', lw=0.8)
ax2.axhline(0, color='gray', lw=0.5)
ax2.set_xlim(10, 40)
ax2.set_ylim(-2, 12)
ax2.set_aspect('equal', adjustable='box')
ax2.grid(True, linestyle=':')
ax2.legend()

plt.show()