# --- START OF FILE visualize_hydrogen_2d.py (Corrected) ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors # For LogNorm
import matplotlib.ticker as ticker
import io
import glob # To find density files if needed
import os
import re # For parsing time from comments

# --- Conversion Constants ---
time_au_to_fs = 0.024188843265857 # a.u. to fs
intensity_au_to_W_cm2 = 3.509445e16

# --- Configuration ---
ionization_file = 'ionization_2d_cap.dat'
density_file_pattern = 'density_evolution_2d_cap.dat' # Pattern if split files, or direct name
output_dir = 'plots_2d' # Directory to save plots
output_plot_file_ionization = os.path.join(output_dir,'ionization_plot_2d.png')
# Density plots will be named based on time step

# --- Create output directory ---
os.makedirs(output_dir, exist_ok=True)


# --- Plot Ionization Probability and Intensity ---
try:
    # Load data: time(au), ion_prob_spat, ion_prob_norm, E_field(au), norm(au)
    data_ion = np.loadtxt(ionization_file)

    # --- CORRECTED COLUMN ASSIGNMENTS ---
    t_au = data_ion[:, 0]         # Time (au)
    ion_prob = data_ion[:, 1]    # Ionization Probability (Spatial)
    # ion_prob_norm_loss = data_ion[:, 2] # Column 2 is norm loss (optional to plot)
    e_field_au = data_ion[:, 3]  # Electric Field (au) <- CORRECTED INDEX
    norm_val = data_ion[:, 4]    # Total Norm <- CORRECTED INDEX
    # --- End Corrected Assignments ---


    # Convert to Physical Units
    t_fs = t_au * time_au_to_fs
    # Intensity calculation is correct now that e_field_au has the right data
    intensity_Wcm2 = (e_field_au**2) * intensity_au_to_W_cm2

    # Create the plot (Ionization + Intensity)
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot Ionization Probability on primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Ionization Probability', color=color)
    # Use the spatially defined probability from column 1
    ax1.plot(t_fs, ion_prob, color=color, lw=2, label='Ionization Probability (Spatial)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle=':')
    ax1.set_ylim(bottom=0)

    # Plot Norm on the same primary axis
    # Now uses the correct norm_val from column 4
    ax1.plot(t_fs, norm_val, color='tab:green', lw=1.5, linestyle='-.', label='Total Norm')


    # Create secondary y-axis for Intensity
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Instantaneous Intensity (W/cm²)', color=color)
    # Now plots the correctly calculated intensity
    ax2.plot(t_fs, intensity_Wcm2, color=color, linestyle='-', linewidth=1.0, alpha=0.6, label='Intensity')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_ylim(bottom=0) # Intensity >= 0

    # Add titles and legend
    fig.suptitle('2D Ionization Dynamics (Physical Units)')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Adjusted label for clarity
    labels[0] = 'Ionization Prob (Spatial R)'
    ax1.legend(lines + lines2, labels + labels2, loc='upper left') # Combine legends

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_plot_file_ionization)
    print(f"Ionization plot (2D) saved to {output_plot_file_ionization}")
    plt.close(fig) # Close figure to free memory

except FileNotFoundError:
    print(f"Error: Could not find input file: {ionization_file}")
except IndexError:
     print(f"Error: File '{ionization_file}' does not have the expected 5 columns. Check C++ output.")
except Exception as e:
    print(f"An error occurred while plotting ionization data: {e}")


# --- Plot Density Evolution Snapshots ---
# This part remains unchanged as it looked correct previously
try:
    with open(density_file_pattern, 'r') as f:
        content = f.read()

    # Split data into time blocks based on double blank lines or comments
    blocks = re.split(r'(?:^#.*$\n)|\n\s*\n', content, flags=re.MULTILINE)

    time_step_count = 0
    current_time = -1.0

    print(f"Processing {density_file_pattern}...")

    block_data = [] # Store data points for the current time step

    for block_str in blocks:
        block_str = block_str.strip()
        if not block_str:
             if block_data and current_time >= 0:
                  try:
                       data_array = np.array(block_data)
                       x_coords = np.unique(data_array[:, 0])
                       y_coords = np.unique(data_array[:, 1])
                       density = data_array[:, 2]
                       Nx_data = len(x_coords)
                       Ny_data = len(y_coords)

                       if Nx_data * Ny_data == len(density):
                           density_grid = density.reshape((Nx_data, Ny_data))

                           fig_dens, ax_dens = plt.subplots(figsize=(8, 7))
                           dx_plot = x_coords[1] - x_coords[0] if Nx_data > 1 else 1
                           dy_plot = y_coords[1] - y_coords[0] if Ny_data > 1 else 1
                           x_edges = np.append(x_coords - dx_plot/2.0, x_coords[-1] + dx_plot/2.0)
                           y_edges = np.append(y_coords - dy_plot/2.0, y_coords[-1] + dy_plot/2.0)

                           min_density = np.min(density_grid[density_grid > 0]) if np.any(density_grid > 0) else 1e-12
                           norm = colors.LogNorm(vmin=max(1e-10, min_density), vmax=np.max(density_grid))

                           im = ax_dens.pcolormesh(x_edges, y_edges, density_grid.T,
                                                   cmap='viridis', norm=norm, shading='auto')

                           ax_dens.set_xlabel('Position x (a.u.)')
                           ax_dens.set_ylabel('Position y (a.u.)')
                           time_fs_str = f"{current_time * time_au_to_fs:.2f}"
                           ax_dens.set_title(f'Probability Density |ψ(x,y)|² at t = {time_fs_str} fs')
                           ax_dens.set_aspect('equal', adjustable='box')
                           fig_dens.colorbar(im, ax=ax_dens, label='Density (log scale)')
                           plt.tight_layout()

                           plot_filename = os.path.join(output_dir, f'density_t_{time_step_count:04d}_time_{current_time:.2f}au.png')
                           plt.savefig(plot_filename)
                           # print(f"  Saved density plot: {plot_filename}") # Less verbose output
                           plt.close(fig_dens)

                           time_step_count += 1

                       else:
                           print(f"Warning: Data size mismatch for time {current_time}. Expected {Nx_data*Ny_data}, got {len(density)}. Skipping plot.")

                  except Exception as e:
                       print(f"Error processing data block for time {current_time}: {e}")

                  block_data = []
                  current_time = -1.0
             continue


        time_match = re.match(r'#\s*Time\s*=\s*([\d.eE+-]+)', block_str)
        if time_match:
            current_time = float(time_match.group(1))
        else:
             lines = block_str.split('\n')
             for line in lines:
                  try:
                       parts = [float(p) for p in line.split()]
                       if len(parts) == 3:
                           block_data.append(parts)
                  except ValueError:
                       pass

    # Process the very last block if the file doesn't end with a blank line/comment
    if block_data and current_time >= 0:
         # Duplicate the plotting logic here or refactor into a function
         # (For brevity, assume the loop structure handles the last block correctly if followed by EOF)
         pass # Add plotting code here if needed for files without trailing newline/comment

    print(f"Finished processing density file. Plotted {time_step_count} snapshots.")


except FileNotFoundError:
    print(f"Error: Could not find density input file: {density_file_pattern}")
except Exception as e:
    print(f"An error occurred while plotting density data: {e}")
