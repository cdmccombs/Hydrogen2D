import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import os
import re
import io

# --- Configuration ---
density_data_file = 'density_evolution_2d_cap.dat'
output_animation_file = 'density_evolution_2d_cap.gif' # Or 'density_evolution_2d.mp4'
animation_fps = 10
animation_dpi = 100

# --- Conversion Constants ---
time_au_to_fs = 2.4188843265857e-17 * 1e15

# --- REVISED Data Reading and Parsing ---
def parse_density_data_revised(filename):
    snapshots = []
    current_time_au = -1.0
    current_data_points = []
    parsed_count = 0

    print("Parsing data file (revised method)...")
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()

                # Check for Time marker
                time_match = re.match(r'#\s*Time\s*=\s*([\d.eE+-]+)', line)

                if time_match:
                    # --- Process the PREVIOUS snapshot's data ---
                    if current_data_points and current_time_au >= 0:
                        try:
                            data_array = np.array(current_data_points)
                            # Check if array is not empty after potential filtering
                            if data_array.size > 0 and data_array.shape[1] == 3:
                                x_coords = np.unique(data_array[:, 0])
                                y_coords = np.unique(data_array[:, 1])
                                density = data_array[:, 2]
                                Nx_data = len(x_coords)
                                Ny_data = len(y_coords)

                                if Nx_data * Ny_data == len(density):
                                    density_grid = density.reshape((Nx_data, Ny_data))
                                    snapshots.append({
                                        'time_au': current_time_au,
                                        'x': x_coords,
                                        'y': y_coords,
                                        'density': density_grid
                                    })
                                    parsed_count += 1
                                else:
                                    print(f"Warning: Data size mismatch for time {current_time_au} AU. Expected {Nx_data*Ny_data}, got {len(density)}. Points found: {len(current_data_points)}")
                            else:
                                 print(f"Warning: No valid data points collected for time {current_time_au} AU before line {line_num+1}.")

                        except ValueError as e:
                             print(f"Error converting data to numbers for time {current_time_au} AU near line {line_num}. Error: {e}")
                        except Exception as e:
                             print(f"Error processing data block for time {current_time_au} AU ending before line {line_num+1}: {e}")

                    # --- Start the NEW snapshot ---
                    current_time_au = float(time_match.group(1))
                    current_data_points = [] # Reset data list for new time
                    # print(f"  Found start of time step: {current_time_au} AU at line {line_num+1}")

                elif current_time_au >= 0 and line: # If we are inside a time block and line is not empty
                    # Assume it's data, attempt to parse
                    try:
                        parts = [float(p) for p in line.split()]
                        if len(parts) == 3:
                            current_data_points.append(parts)
                    except ValueError:
                        # Ignore lines that are not 3 floats within a data block
                        # print(f"  Ignoring non-data line {line_num+1}: '{line}'")
                        pass

            # --- Process the VERY LAST snapshot after EOF ---
            if current_data_points and current_time_au >= 0:
                try:
                    data_array = np.array(current_data_points)
                    if data_array.size > 0 and data_array.shape[1] == 3:
                        x_coords = np.unique(data_array[:, 0])
                        y_coords = np.unique(data_array[:, 1])
                        density = data_array[:, 2]
                        Nx_data = len(x_coords)
                        Ny_data = len(y_coords)

                        if Nx_data * Ny_data == len(density):
                            density_grid = density.reshape((Nx_data, Ny_data))
                            snapshots.append({
                                'time_au': current_time_au,
                                'x': x_coords,
                                'y': y_coords,
                                'density': density_grid
                            })
                            parsed_count += 1
                        else:
                            print(f"Warning: Data size mismatch for final time step {current_time_au} AU. Expected {Nx_data*Ny_data}, got {len(density)}. Points found: {len(current_data_points)}")
                    else:
                        print(f"Warning: No valid data points collected for final time step {current_time_au} AU.")
                except ValueError as e:
                     print(f"Error converting data to numbers for final time step {current_time_au} AU. Error: {e}")
                except Exception as e:
                    print(f"Error processing final data block for time {current_time_au} AU: {e}")


    except FileNotFoundError:
        print(f"Error: Data file not found: {filename}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during file reading: {e}")
        return []


    print(f"Parsed {parsed_count} snapshots using revised method.")
    # Sort snapshots by time just in case
    snapshots.sort(key=lambda s: s['time_au'])
    return snapshots

# --- Main Animation Logic (Keep the rest of the script the same) ---
# ... (Use the new parsing function)
snapshots = parse_density_data_revised(density_data_file)

if not snapshots:
    print("No data found or parsed. Exiting.")
    exit()

# ... (rest of the plotting and animation code remains unchanged) ...

# Determine global min/max density for consistent color scale
all_densities = np.concatenate([s['density'].ravel() for s in snapshots])
min_density = np.min(all_densities[all_densities > 0]) if np.any(all_densities > 0) else 1e-12
max_density = np.max(all_densities) if np.any(all_densities > 0) else 1.0
print(f"Global density range (for log scale): min_positive ~ {min_density:.2e}, max = {max_density:.2e}")

# Setup the figure and axes
fig, ax = plt.subplots(figsize=(8, 7))
x_coords = snapshots[0]['x']
y_coords = snapshots[0]['y']
dx_plot = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1
dy_plot = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1
x_edges = np.append(x_coords - dx_plot/2.0, x_coords[-1] + dx_plot/2.0)
y_edges = np.append(y_coords - dy_plot/2.0, y_coords[-1] + dy_plot/2.0)
norm = colors.LogNorm(vmin=max(1e-10, min_density * 0.1), vmax=max_density)
initial_density = snapshots[0]['density']
im = ax.pcolormesh(x_edges, y_edges, initial_density.T, cmap='viridis', norm=norm, shading='auto')
ax.set_xlabel('Position x (a.u.)')
ax.set_ylabel('Position y (a.u.)')
ax.set_aspect('equal', adjustable='box')
fig.colorbar(im, ax=ax, label='Probability Density |ψ|² (log scale)')
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def init():
    first_snapshot = snapshots[0]
    im.set_array(first_snapshot['density'].T.ravel())
    time_fs = first_snapshot['time_au'] * time_au_to_fs
    time_text.set_text(f'Time = {time_fs:.2f} fs')
    return im, time_text

def update(frame):
    snapshot = snapshots[frame]
    density_grid = snapshot['density']
    time_fs = snapshot['time_au'] * time_au_to_fs
    im.set_array(density_grid.T.ravel())
    time_text.set_text(f'Time = {time_fs:.2f} fs')
    if frame % 20 == 0: print(f"Generating frame {frame+1}/{len(snapshots)}")
    return im, time_text

num_frames = len(snapshots)
ani = animation.FuncAnimation(fig, update, frames=num_frames,
                              init_func=init, blit=True, interval=1000/animation_fps)

# ... (Saving logic remains the same) ...
writer = None
if output_animation_file.endswith('.gif'):
    try: writer = animation.PillowWriter(fps=animation_fps)
    except Exception:
         print("PillowWriter failed. Trying ImageMagick...")
         try: writer = animation.ImageMagickWriter(fps=animation_fps)
         except Exception as e: print(f"ImageMagickWriter failed: {e}")
elif output_animation_file.endswith('.mp4'):
    try: writer = animation.FFMpegWriter(fps=animation_fps)
    except Exception as e: print(f"FFMpegWriter failed: {e}")

if writer:
    print(f"Saving animation to {output_animation_file}...")
    try:
        ani.save(output_animation_file, writer=writer, dpi=animation_dpi)
        print("Animation saved successfully.")
    except Exception as e:
        print(f"\nError saving animation: {e}")
        print("Ensure backend (ffmpeg or imagemagick/pillow) is installed.")
else:
    print("\nCould not find a suitable animation writer.")

plt.close(fig)
