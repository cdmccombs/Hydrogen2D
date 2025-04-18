#include <iostream>
#include <complex>
#include <vector>
#include <fstream>
#include <cmath>
#include <fftw3.h>
#include <stdexcept> // For exceptions
#include <cstring>   // For memcpy in energy calculation

// Define constants (atomic units)
const double PI = 3.14159265358979323846;
const std::complex<double> I(0, 1);

// Helper to get flattened index (row-major)
inline int flatten_index(int i, int j, int Ny) {
    return i * Ny + j;
}

class HydrogenSimulator2D {
private:
    // Simulation parameters
    int Nx, Ny;             // Number of grid points in x and y
    int N;                  // Total grid points (Nx * Ny)
    double dx, dy;          // Grid spacing
    double dt;              // Time step
    double xmin, xmax;      // Grid boundaries in x
    double ymin, ymax;      // Grid boundaries in y
    double a;               // Soft-Coulomb parameter

    // System state
    std::vector<std::complex<double>> psi;  // Wavefunction (flattened 1D)
    std::vector<double> x;                  // Position grid (x-dimension)
    std::vector<double> y;                  // Position grid (y-dimension)
    std::vector<double> V;                  // REAL Potential (Coulomb only) (flattened 1D)

    // --- CAP Parameters ---
    double cap_start_x;     // Absolute distance from origin where CAP starts in x
    double cap_start_y;     // Absolute distance from origin where CAP starts in y
    double cap_eta;         // Strength of the CAP
    std::vector<double> W_cap; // Imaginary absorbing potential W(x,y) >= 0

    // FFTW plans and arrays
    fftw_complex *fftw_in, *fftw_out;
    fftw_plan forward_plan, backward_plan;

    // Laser field parameters
    double E0;              // Field amplitude
    double omega;           // Field frequency
    double cycles;          // Number of laser cycles
    double phase;           // CEP phase

    // --- Removed Absorbing Boundary Mask ---
    // std::vector<double> absorb_mask;

public:
    HydrogenSimulator2D(int gridPointsX, int gridPointsY,
                        double gridSpacingX, double gridSpacingY,
                        double timeStep, double softParameter,
                        double fieldAmp, double fieldFreq,
                        double numCycles, double cepPhase,
                        // --- Add CAP parameters to constructor ---
                        double capStartX, double capStartY, double capStrength)
        : Nx(gridPointsX), Ny(gridPointsY),
          dx(gridSpacingX), dy(gridSpacingY), dt(timeStep), a(softParameter),
          E0(fieldAmp), omega(fieldFreq), cycles(numCycles), phase(cepPhase),
          // --- Initialize CAP parameters ---
          cap_start_x(capStartX), cap_start_y(capStartY), cap_eta(capStrength)
    {
        N = Nx * Ny;
        if (N <= 0) {
            throw std::runtime_error("Grid dimensions must be positive.");
        }

        // Initialize grid boundaries
        // Use floor/ceil for robustness if Nx/Ny can be odd
        xmin = -std::floor(Nx / 2.0) * dx;
        xmax = std::floor((Nx - 1.0) / 2.0) * dx;
        ymin = -std::floor(Ny / 2.0) * dy;
        ymax = std::floor((Ny - 1.0) / 2.0) * dy;


        // Allocate vectors
        x.resize(Nx);
        y.resize(Ny);
        V.resize(N);
        psi.resize(N);
        W_cap.resize(N); // Allocate space for CAP potential

        // Set up position grids
        for (int i = 0; i < Nx; ++i) {
            x[i] = xmin + i * dx;
        }
        for (int j = 0; j < Ny; ++j) {
            y[j] = ymin + j * dy;
        }

        // Check CAP parameters validity
        if (cap_start_x < 0 || cap_start_y < 0) {
            throw std::runtime_error("CAP start distances must be non-negative.");
        }
         if (cap_start_x >= (xmax - xmin)/2.0 || cap_start_y >= (ymax - ymin)/2.0 ) {
             // Check if start is beyond the edge, considering symmetry
             // A more robust check: Ensure CAP starts *before* the absolute max coordinate
             if (cap_start_x >= std::abs(xmax) || cap_start_x >= std::abs(xmin) ||
                 cap_start_y >= std::abs(ymax) || cap_start_y >= std::abs(ymin))
             {
                throw std::runtime_error("CAP start position must be strictly inside the grid boundaries.");
             }
         }
         if (cap_eta < 0) {
             throw std::runtime_error("CAP strength eta must be non-negative.");
         }


        // Set up REAL potential (V) and CAP potential (W_cap)
        for (int i = 0; i < Nx; ++i) {
            double Wx = 0.0;
            if (std::abs(x[i]) > cap_start_x) {
                double dx_cap = std::abs(x[i]) - cap_start_x;
                Wx = cap_eta * dx_cap * dx_cap; // Quadratic CAP in x
            }

            for (int j = 0; j < Ny; ++j) {
                double Wy = 0.0;
                if (std::abs(y[j]) > cap_start_y) {
                     double dy_cap = std::abs(y[j]) - cap_start_y;
                     Wy = cap_eta * dy_cap * dy_cap; // Quadratic CAP in y
                }

                int idx = flatten_index(i, j, Ny);
                // Store the REAL Coulomb potential
                V[idx] = -1.0 / std::sqrt(x[i]*x[i] + y[j]*y[j] + a*a);
                // Store the CAP potential W(x,y) >= 0
                W_cap[idx] = Wx + Wy;

                // --- Removed absorbing mask calculation ---
            }
        }

        // Initialize FFTW (use N = Nx * Ny)
        fftw_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        fftw_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
        if (!fftw_in || !fftw_out) {
             throw std::runtime_error("FFTW malloc failed.");
        }

        // Create 2D plans
        forward_plan = fftw_plan_dft_2d(Nx, Ny, fftw_in, fftw_out, FFTW_FORWARD, FFTW_MEASURE);
        backward_plan = fftw_plan_dft_2d(Nx, Ny, fftw_in, fftw_out, FFTW_BACKWARD, FFTW_MEASURE);
        if (!forward_plan || !backward_plan) {
             throw std::runtime_error("FFTW plan creation failed.");
        }

        // Initialize wavefunction (ground state using ONLY real potential V)
        initializeGroundState();
    }

    ~HydrogenSimulator2D() {
        // Clean up FFTW resources
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(backward_plan);
        fftw_free(fftw_in);
        fftw_free(fftw_out);
    }

    // Forward 2D FFT
    void forwardFFT(std::vector<std::complex<double>>& data) {
        // Copy data to FFTW input buffer
        memcpy(fftw_in, data.data(), N * sizeof(fftw_complex));
        // Execute plan
        fftw_execute(forward_plan);
        // Copy result back from FFTW output buffer
        memcpy(data.data(), fftw_out, N * sizeof(fftw_complex));
    }

    // Backward 2D FFT
    void backwardFFT(std::vector<std::complex<double>>& data) {
         // Copy data to FFTW input buffer
        memcpy(fftw_in, data.data(), N * sizeof(fftw_complex));
        // Execute plan
        fftw_execute(backward_plan);
        // Copy result back and normalize
        double norm_factor = 1.0 / static_cast<double>(N);
         for (int i = 0; i < N; ++i) {
            data[i] = std::complex<double>(fftw_out[i][0], fftw_out[i][1]) * norm_factor;
        }
    }

    // Calculate ground state using imaginary time propagation (Uses REAL potential V only)
    void initializeGroundState() {
        // Start with a 2D Gaussian wavefunction
        double sigma = 1.0;
        double norm = 0.0;
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                 int idx = flatten_index(i, j, Ny);
                 psi[idx] = std::exp(-(x[i]*x[i] + y[j]*y[j])/(2.0*sigma*sigma));
                 norm += std::norm(psi[idx]) * dx * dy;
            }
        }
        // Normalize
        double norm_sqrt = std::sqrt(norm);
        for (int idx = 0; idx < N; ++idx) {
            psi[idx] /= norm_sqrt;
        }

        // Imaginary time propagation
        double imag_dt = 0.05;
        std::cout << "Starting 2D ground state calculation..." << std::endl;

        for (int step = 0; step < 1000; ++step) {
            // Half step potential (REAL potential V only)
            for (int idx = 0; idx < N; ++idx) {
                psi[idx] *= std::exp(-V[idx] * imag_dt / 2.0); // NO W_cap here
            }

            // Kinetic step
            forwardFFT(psi);
            for (int i = 0; i < Nx; ++i) {
                 double kx;
                 // Standard FFTW k-vector definition (adjust if using different FFT library/convention)
                 if (i <= Nx/2) kx = 2.0 * PI * i / (Nx * dx);
                 else           kx = 2.0 * PI * (i - Nx) / (Nx * dx);

                 for (int j = 0; j < Ny; ++j) {
                      double ky;
                      if (j <= Ny/2) ky = 2.0 * PI * j / (Ny * dy);
                      else           ky = 2.0 * PI * (j - Ny) / (Ny * dy);

                      int idx = flatten_index(i, j, Ny);
                      double kinetic_factor = (kx*kx + ky*ky) / 2.0;
                      psi[idx] *= std::exp(-kinetic_factor * imag_dt); // Imaginary time -> real exponential decay
                 }
            }
            backwardFFT(psi);

            // Half step potential (REAL potential V only)
            for (int idx = 0; idx < N; ++idx) {
                psi[idx] *= std::exp(-V[idx] * imag_dt / 2.0); // NO W_cap here
            }

            // Normalize
            norm = 0.0;
            for (int idx = 0; idx < N; ++idx) {
                norm += std::norm(psi[idx]) * dx * dy;
            }
            norm_sqrt = std::sqrt(norm);
            if (norm_sqrt > 1e-15) {
                for (int idx = 0; idx < N; ++idx) {
                    psi[idx] /= norm_sqrt;
                }
            } else {
                 std::cerr << "Warning: Wavefunction norm close to zero during imaginary time prop." << std::endl;
                 // Could add logic to re-initialize if norm gets too small
            }

            // Occasionally calculate energy (using the real Hamiltonian)
            if (step % 100 == 0 || step == 999) {
                double energy = calculateEnergy_alternative(); // Uses H_physical
                std::cout << "Step " << step << ", Energy = " << energy
                          << " a.u. (" << energy * 27.2114 << " eV)" << std::endl;
            }
        }

        double final_energy = calculateEnergy_alternative();
        std::cout << "2D Ground state energy: " << final_energy
                  << " a.u. (" << final_energy * 27.2114 << " eV)" << std::endl;
    }

    // Calculate total energy using <psi|H_physical|psi> = <psi|T|psi> + <psi|V|psi>
    // NOTE: This energy is NOT conserved when the CAP is active in real time prop.
    double calculateEnergy_alternative() {
        // Potential Energy: <psi|V|psi> (using the REAL potential V)
        double potential_energy = 0.0;
        for (int idx = 0; idx < N; ++idx) {
            potential_energy += (std::conj(psi[idx]) * V[idx] * psi[idx]).real() * dx * dy;
        }

        // Kinetic Energy: <psi|T|psi>
        std::vector<std::complex<double>> T_psi = psi; // Copy psi
        forwardFFT(T_psi); // Transform to k-space

        // Apply kinetic operator (kx^2+ky^2)/2 in k-space
        for (int i = 0; i < Nx; ++i) {
            double kx;
            if (i <= Nx/2) kx = 2.0 * PI * i / (Nx * dx);
            else           kx = 2.0 * PI * (i - Nx) / (Nx * dx);
            for (int j = 0; j < Ny; ++j) {
                 double ky;
                 if (j <= Ny/2) ky = 2.0 * PI * j / (Ny * dy);
                 else           ky = 2.0 * PI * (j - Ny) / (Ny * dy);
                 int idx = flatten_index(i, j, Ny);
                 double kinetic_factor = (kx*kx + ky*ky) / 2.0;
                 T_psi[idx] *= kinetic_factor;
            }
        }
        backwardFFT(T_psi); // Transform back T|psi> to position space

        // Calculate inner product <psi|T|psi> = sum conj(psi_i) * (T_psi)_i * dx * dy
        double kinetic_energy = 0.0;
        for (int idx = 0; idx < N; ++idx) {
            kinetic_energy += (std::conj(psi[idx]) * T_psi[idx]).real() * dx * dy;
        }

        return potential_energy + kinetic_energy;
    }

    // Calculate electric field at time t
    double electricField(double t) {
        double env = 0.0;
        double t_duration = (cycles > 0 && omega > 0) ? (2.0 * PI * cycles / omega) : 0.0;

        if (t_duration > 0 && t >= 0 && t < t_duration) {
            env = std::sin(PI * t / t_duration); // sin^2 envelope
            env *= env;
            return E0 * env * std::cos(omega * t + phase);
        }
        return 0.0;
    }

    // Propagate wavefunction for one time step (Split-Operator with CAP)
    void propagateStep(double t) {
        // --- Half step potential + laser + CAP ---
        double E_t = electricField(t);
        for (int i = 0; i < Nx; ++i) {
            double laser_term = x[i] * E_t;
            for (int j = 0; j < Ny; ++j) {
                int idx = flatten_index(i, j, Ny);
                // Real part of potential for phase evolution
                double V_real_t = V[idx] + laser_term;
                // Combine phase factor and CAP damping factor
                // exp(-i*H*dt/2) approx exp(-i*V_real*dt/2) * exp(-W*dt/2)
                std::complex<double> potential_phase_factor = std::exp(-I * V_real_t * dt / 2.0);
                double cap_damping_factor = std::exp(-W_cap[idx] * dt / 2.0); // W_cap >= 0
                psi[idx] *= potential_phase_factor * cap_damping_factor;
            }
        }

        // --- Full step kinetic (Unchanged) ---
        forwardFFT(psi);
        for (int i = 0; i < Nx; ++i) {
             double kx;
             if (i <= Nx/2) kx = 2.0 * PI * i / (Nx * dx);
             else           kx = 2.0 * PI * (i - Nx) / (Nx * dx);
             for (int j = 0; j < Ny; ++j) {
                  double ky;
                  if (j <= Ny/2) ky = 2.0 * PI * j / (Ny * dy);
                  else           ky = 2.0 * PI * (j - Ny) / (Ny * dy);
                  int idx = flatten_index(i, j, Ny);
                  double kinetic_factor = (kx*kx + ky*ky) / 2.0;
                  // exp(-i*T*dt)
                  psi[idx] *= std::exp(-I * kinetic_factor * dt);
             }
        }
        backwardFFT(psi);

        // --- Second Half step potential + laser (at t+dt) + CAP ---
        double E_t_plus_dt = electricField(t + dt);
        for (int i = 0; i < Nx; ++i) {
            double laser_term = x[i] * E_t_plus_dt;
            for (int j = 0; j < Ny; ++j) {
                int idx = flatten_index(i, j, Ny);
                // Real part of potential for phase evolution at t+dt
                double V_real_t_plus_dt = V[idx] + laser_term;
                 // Combine phase factor and CAP damping factor
                std::complex<double> potential_phase_factor = std::exp(-I * V_real_t_plus_dt * dt / 2.0);
                double cap_damping_factor = std::exp(-W_cap[idx] * dt / 2.0); // Same W_cap
                psi[idx] *= potential_phase_factor * cap_damping_factor;
            }
        }
    }

    // --- Remove applyAbsorbingBoundary method ---
    // void applyAbsorbingBoundary() {
    //     // No longer needed, CAP is part of the propagation step
    // }

    // Calculate ionization probability (outside a circular region)
    // This definition remains the same (based on spatial location)
    double calculateIonization(double bound_radius_sq) { // Pass squared radius
        double bound_prob = 0.0;
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                if ((x[i]*x[i] + y[j]*y[j]) < bound_radius_sq) {
                    int idx = flatten_index(i, j, Ny);
                    bound_prob += std::norm(psi[idx]) * dx * dy; // |psi|^2 * dV
                }
            }
        }
        // Ionization is the probability *not* in the bound region.
        // Note: With CAP, the total norm is NOT conserved.
        // 1.0 - bound_prob represents the probability currently outside R,
        // *plus* any probability already absorbed by the CAP.
        // This might be what you want (total probability that left the R region).
        // Alternatively, calculate norm N(t) and define ionization as 1 - N(t).
        // Let's keep the spatial definition for now.
        return 1.0 - bound_prob;
    }

    // Run simulation
    void run(int timesteps, double bound_radius) {
        std::ofstream outFile("ionization_2d_cap.dat");
        std::ofstream densityFile("density_evolution_2d_cap.dat");

        double t = 0.0;
        double initial_energy = calculateEnergy_alternative(); // Energy of H_physical
        std::cout << "Initial 2D energy (H_physical): " << initial_energy << " a.u. ("
                  << initial_energy * 27.2114 << " eV)" << std::endl;
        double initial_norm = 0.0;
        for(int idx=0; idx<N; ++idx) initial_norm += std::norm(psi[idx])*dx*dy;
         std::cout << "Initial Norm: " << initial_norm << std::endl;


        double bound_radius_sq = bound_radius * bound_radius;
        int density_save_interval = 500; // Save density less often
        int density_downsample = 4;      // Save only every Nth point

        for (int step = 0; step < timesteps; ++step) {
            propagateStep(t);
            // --- No call to applyAbsorbingBoundary() needed ---

            // Calculate observables periodically
            if (step % 10 == 0) {
                 // Calculate norm to check decay due to CAP
                 double current_norm = 0;
                 for(int idx=0; idx<N; ++idx) current_norm += std::norm(psi[idx])*dx*dy;

                 // Calculate ionization based on spatial region
                 double ion_prob_spatial = calculateIonization(bound_radius_sq);

                 // Alternative: Ionization as norm loss
                 double ion_prob_norm_loss = initial_norm - current_norm; // Assuming initial_norm ~ 1

                outFile << t << "\t"
                        << ion_prob_spatial << "\t"    // Prob outside R
                        << ion_prob_norm_loss << "\t"  // Prob absorbed
                        << electricField(t) << "\t"
                        << current_norm << std::endl;

                // Save density snapshot (infrequently and downsampled)
                if (step % density_save_interval == 0) {
                    std::cout << "Step " << step << ", Time: " << t << " a.u."
                              << ", Ion(space): " << ion_prob_spatial
                              << ", Norm: " << current_norm << std::endl;
                    densityFile << "# Time = " << t << std::endl;
                    for (int i = 0; i < Nx; i += density_downsample) {
                        for (int j = 0; j < Ny; j += density_downsample) {
                            int idx = flatten_index(i, j, Ny);
                            densityFile << x[i] << "\t" << y[j] << "\t"
                                       << std::norm(psi[idx]) << std::endl;
                        }
                         densityFile << std::endl;
                    }
                     densityFile << std::endl;
                }
            }

            t += dt;
        }

        outFile.close();
        densityFile.close();
        std::cout << "Final Norm: " << calculateNorm() << std::endl; // Helper function added below
    }

    // Helper function to calculate current norm
    double calculateNorm() {
        double current_norm_sq = 0.0;
        for (int idx = 0; idx < N; ++idx) {
            current_norm_sq += std::norm(psi[idx]);
        }
        return current_norm_sq * dx * dy; // Don't forget volume element
    }

}; // End class HydrogenSimulator2D

int main() {
    // --- Simulation Parameters (Adjust Carefully!) ---
    int Nx = 1024;           // Grid points X (640)
    int Ny = 1024;           // Grid points Y (640)
    double dx = 0.4;           // Grid spacing X (a.u.)
    double dy = 0.4;           // Grid spacing Y (a.u.)
    double dt = 0.03;           // Time step (a.u.)
    double a = 0.79837;             // Soft-Coulomb parameter

    // Laser parameters
    double E0 = 0.134085;           // Field amplitude (a.u.)
    double omega = 0.057;       // Frequency (a.u. ~ 800 nm)
    double cycles = 3.0;        // Pulse duration
    double phase = 0.0;         // CEP

    // Analysis parameters
    double bound_radius = 15.0; // Radius (a.u.) to consider "bound" spatially
    int timesteps = 12000;      // Total simulation steps

    // --- CAP Parameters (NEED TUNING!) ---
    double grid_radius_x = (Nx / 2.0) * dx; // Approx edge distance ~80
    double grid_radius_y = (Ny / 2.0) * dy; // Approx edge distance ~80
    double cap_start_x = grid_radius_x * 0.75;  // Example: Start CAP at 70% of grid radius
    double cap_start_y = grid_radius_y * 0.75;  // Example: Start CAP at 70% of grid radius
    double cap_eta = 0.05;     // Example: Strength parameter (Start here, tune based on results)

    std::cout << "2D Hydrogen in laser field simulation with CAP" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Grid: " << Nx << "x" << Ny << ", dx=" << dx << ", dy=" << dy << std::endl;
    std::cout << "Grid extends approx to +/- " << grid_radius_x << " (x), +/- " << grid_radius_y << " (y) a.u." << std::endl;
    std::cout << "Soft-Coulomb parameter a = " << a << std::endl;
    std::cout << "Laser Intensity approx " << E0*E0*3.51e16 << " W/cm^2" << std::endl;
    std::cout << "Laser wavelength approx " << 45.563/omega << " nm" << std::endl;
    std::cout << "Pulse duration = " << cycles << " cycles (" << (2.0*PI*cycles/omega) * 24.19e-3 << " fs)" << std::endl;
    std::cout << "Timesteps = " << timesteps << ", dt=" << dt << std::endl;
    std::cout << "Using Quadratic CAP:" << std::endl;
    std::cout << "  Start X: +/- " << cap_start_x << " a.u." << std::endl;
    std::cout << "  Start Y: +/- " << cap_start_y << " a.u." << std::endl;
    std::cout << "  Eta: " << cap_eta << std::endl;

    try {
        HydrogenSimulator2D simulator(Nx, Ny, dx, dy, dt, a,
                                      E0, omega, cycles, phase,
                                      cap_start_x, cap_start_y, cap_eta); // Pass CAP params
        simulator.run(timesteps, bound_radius);

        std::cout << "Simulation complete." << std::endl;
        std::cout << "Data saved to ionization_2d_cap.dat and density_evolution_2d_cap.dat" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
         std::cerr << "An unknown error occurred." << std::endl;
         return 1;
    }

    return 0;
}
