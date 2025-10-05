"""
Demo script for orbit propagation using aholinch/sgp4 library
from: https://github.com/aholinch/sgp4

Requirements:
- TLE.py, SGP4.py, and ElsetRec.py from the repository
- matplotlib for plotting
"""

import matplotlib.pyplot as plt
import numpy as np
from TLE import TLE

prop_days = 1
filepath = "../example.tle"


def load_tle_from_file(filename):
    """
    Load TLEs from a file.
    Returns a list of TLE objects.
    """
    tles = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Process TLEs in pairs (line 1 and line 2)
    i = 0
    while i < len(lines) - 1:
        line1 = lines[i].strip()
        line2 = lines[i + 1].strip()
        
        # Check if these are valid TLE lines
        if line1.startswith('1 ') and line2.startswith('2 '):
            try:
                tle = TLE(line1, line2)
                tles.append(tle)
                i += 2
            except Exception as e:
                print(f"Warning: Could not parse TLE at line {i}: {e}")
                i += 1
        else:
            i += 1
    
    return tles


def propagate_orbit(tle, n_days=1, n_points=10000):
    """
    Propagate orbit for a given TLE.
    
    Args:
        tle: TLE object
        n_days: Number of days to propagate
        n_points: Number of time points to compute
        
    Returns:
        positions: numpy array of shape (n_points, 3) with x, y, z positions in km
        velocities: numpy array of shape (n_points, 3) with vx, vy, vz in km/s
        times: numpy array of time points in minutes since epoch
    """
    # Generate time array in minutes since epoch
    times = np.linspace(0, n_days * 24 * 60, n_points)
    
    positions = []
    velocities = []
    
    for t in times:
        try:
            # getRV returns [[x, y, z], [vx, vy, vz]] in km and km/s
            rv = tle.getRV(t)
            positions.append(rv[0])
            velocities.append(rv[1])
        except Exception as e:
            print(f"Warning: Propagation failed at t={t:.2f} min: {e}")
            # Use NaN for failed points
            positions.append([np.nan, np.nan, np.nan])
            velocities.append([np.nan, np.nan, np.nan])
    
    return np.array(positions), np.array(velocities), times


def plot_orbit(positions, label='Orbit', color='lightcoral'):
    """
    Plot 3D orbit trajectory.
    
    Args:
        positions: numpy array of shape (n_points, 3) with x, y, z positions
        label: Label for the plot
        color: Color for the orbit line
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot orbit
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            color=color, label=label, linewidth=0.8)
    
    # Plot Earth as a sphere
    u = np.linspace(0, 2 * np.pi, 50) # longitude-like
    v = np.linspace(0, np.pi, 50) # latitude-like
    earth_radius = 6378.137  # km
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    earth_surface = ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
    earth_surface.set_label('Earth')
    
    # Set labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Satellite Orbit')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()


def main():
    # Load all TLEs from file
    tles = load_tle_from_file(filepath)
    
    if not tles:
        print("Error: No TLEs found in file")
        return
    
    # Extract the first TLE
    my_tle = tles[0]
    
    # Print TLE information
    print(f"Satellite Catalog Number: {my_tle.objectID}")
    print(f"International Designator: {my_tle.intlid}")
    print(f"Epoch Year: {my_tle.rec.epochyr}")
    print(f"Epoch Days: {my_tle.rec.epochdays:.8f}")
    print(f"Inclination: {my_tle.incDeg:.4f} deg")
    print(f"RAAN: {my_tle.raanDeg:.4f} deg")
    print(f"Eccentricity: {my_tle.ecc:.7f}")
    print(f"Argument of Perigee: {my_tle.argpDeg:.4f} deg")
    print(f"Mean Anomaly: {my_tle.maDeg:.4f} deg")
    print(f"Mean Motion: {my_tle.n:.8f} rev/day")
    
    # Calculate orbital period
    period_minutes = float(24*60) / my_tle.n  # minutes per revolution
    print(f"Orbital Period: {period_minutes:.2f} minutes ({period_minutes/60:.2f} hours)")
    
    # Propagate for n day(s)
    print(f"\nPropagating orbit for {prop_days} day(s)...")
    positions, velocities, times = propagate_orbit(my_tle, n_days=prop_days, n_points=10000)
    
    # Check for propagation errors
    if my_tle.sgp4Error != 0:
        print(f"Warning: SGP4 error code {my_tle.sgp4Error}")
    
    # Plot the orbit
    plot_orbit(positions, 
               label=f'SATCAT n°: {my_tle.objectID}',
               color='lightcoral')
    
    plt.show()


if __name__ == "__main__":
    main()