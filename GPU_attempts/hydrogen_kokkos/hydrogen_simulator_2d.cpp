#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <string>

// Kokkos headers
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

// Define constants (atomic units)
const double PI = 3.14159265358979323846;

// Type aliases for clarity
using complex_type = Kokkos::complex<double>;
using complex_view_1d = Kokkos::View<complex_type*>;
using complex_view_2d = Kokkos::View<complex_type**>;
using real_view_1d = Kokkos::View<double*>;
using real_view_2d = Kokkos::View<double**>;

// Host mirrors for data transfer
using complex_host_view_1d = complex_view_1d::HostMirror;
using complex_host_view_2d = complex_view_2d::HostMirror;
using real_host_view_1d = real_view_1d::HostMirror;
using real_host_view_2d = real_view_2d::HostMirror;

// Forward declaration of cuFFT wrapper functions
extern "C" {
void forward_fft_2d(void* data, int nx, int ny);
void backward_fft_2d(void* data, int nx, int ny);
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

    // System state - Kokkos Views
    complex_view_2d d_psi;        // Wavefunction on device (2D layout)
    complex_view_1d d_psi_flat;   // Flattened wavefunction for FFT
    real_view_1d d_x;             // Position grid (x-dimension)
    real_view_1d d_y;             // Position grid (y-dimension)
    real_view_2d d_V;             // Potential (Coulomb only)
    
    // Host mirrors for data transfer and I/O
    complex_host_view_2d h_psi;
    complex_host_view_1d h_psi_flat;
    real_host_view_1d h_x;
    real_host_view_1d h_y;
    real_host_view_2d h_V;

    // CAP Parameters
    double cap_start_x;     // Absolute distance from origin where CAP starts in x
    double cap_start_y;     // Absolute distance from origin where CAP starts in y
    double cap_eta;         // Strength of the CAP
    real_view_2d d_W_cap;   // Imaginary absorbing potential W(x,y) >= 0
    real_host_view_2d h_W_cap;
    
    // Laser field parameters
    double E0;              // Field amplitude
    double omega;           // Field frequency
    double cycles;          // Number of laser cycles
    double phase;           // CEP phase

public:
    HydrogenSimulator2D(int gridPointsX, int gridPointsY,
                        double gridSpacingX, double gridSpacingY,
                        double timeStep, double softParameter,
                        double fieldAmp, double fieldFreq,
                        double numCycles, double cepPhase,
                        double capStartX, double capStartY, double capStrength)
        : Nx(gridPointsX), Ny(gridPointsY),
          dx(gridSpacingX), dy(gridSpacingY), dt(timeStep), a(softParameter),
          E0(fieldAmp), omega(fieldFreq), cycles(numCycles), phase(cepPhase),
          cap_start_x(capStartX), cap_start_y(capStartY), cap_eta(capStrength)
    {
        N = Nx * Ny;
        if (N <= 0) {
            throw std::runtime_error("Grid dimensions must be positive.");
        }

        // Initialize grid boundaries
        xmin = -std::floor(Nx / 2.0) * dx;
        xmax = std::floor((Nx - 1.0) / 2.0) * dx;
        ymin = -std::floor(Ny / 2.0) * dy;
        ymax = std::floor((Ny - 1.0) / 2.0) * dy;

        // Allocate Kokkos Views
        d_psi = complex_view_2d("psi", Nx, Ny);
        d_psi_flat = complex_view_1d("psi_flat", N);
        d_x = real_view_1d("x", Nx);
        d_y = real_view_1d("y", Ny);
        d_V = real_view_2d("V", Nx, Ny);
        d_W_cap = real_view_2d("W_cap", Nx, Ny);

        // Create host mirrors
        h_psi = Kokkos::create_mirror_view(d_psi);
        h_psi_flat = Kokkos::create_mirror_view(d_psi_flat);
        h_x = Kokkos::create_mirror_view(d_x);
        h_y = Kokkos::create_mirror_view(d_y);
        h_V = Kokkos::create_mirror_view(d_V);
        h_W_cap = Kokkos::create_mirror_view(d_W_cap);

        // Check CAP parameters validity
        if (cap_start_x < 0 || cap_start_y < 0) {
            throw std::runtime_error("CAP start distances must be non-negative.");
        }
        if (cap_start_x >= std::abs(xmax) || cap_start_x >= std::abs(xmin) ||
            cap_start_y >= std::abs(ymax) || cap_start_y >= std::abs(ymin)) {
            throw std::runtime_error("CAP start position must be strictly inside the grid boundaries.");
        }
        if (cap_eta < 0) {
            throw std::runtime_error("CAP strength eta must be non-negative.");
        }

        // Initialize position grids on host
        for (int i = 0; i < Nx; ++i) {
            h_x(i) = xmin + i * dx;
        }
        for (int j = 0; j < Ny; ++j) {
            h_y(j) = ymin + j * dy;
        }

        // Copy position grids to device
        Kokkos::deep_copy(d_x, h_x);
        Kokkos::deep_copy(d_y, h_y);

        // Initialize potential and CAP on device
        initializePotentialAndCAP();

        // Initialize wavefunction (ground state)
        initializeGroundState();
    }

    // Destructor
    ~HydrogenSimulator2D() {
        // Kokkos Views are automatically cleaned up
    }

    // Initialize potential and CAP on device
    void initializePotentialAndCAP() {
        // Lambda to compute both potential and CAP on device
        Kokkos::parallel_for("init_potential_cap", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                // Get position values
                double xi = d_x(i);
                double yj = d_y(j);
                
                // Calculate Coulomb potential
                d_V(i, j) = -1.0 / std::sqrt(xi*xi + yj*yj + a*a);
                
                // Calculate CAP
                double Wx = 0.0;
                if (std::abs(xi) > cap_start_x) {
                    double dx_cap = std::abs(xi) - cap_start_x;
                    Wx = cap_eta * dx_cap * dx_cap; // Quadratic CAP in x
                }
                
                double Wy = 0.0;
                if (std::abs(yj) > cap_start_y) {
                    double dy_cap = std::abs(yj) - cap_start_y;
                    Wy = cap_eta * dy_cap * dy_cap; // Quadratic CAP in y
                }
                
                d_W_cap(i, j) = Wx + Wy;
            }
        );
        
        // Copy potential and CAP to host for possible output/visualization
        Kokkos::deep_copy(h_V, d_V);
        Kokkos::deep_copy(h_W_cap, d_W_cap);
    }

    // Helper to flatten 2D wavefunction to 1D for FFT processing
    void flatten_wavefunction() {
        Kokkos::parallel_for("flatten_wavefunction", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                d_psi_flat(i * Ny + j) = d_psi(i, j);
            }
        );
    }

    // Helper to unflatten 1D FFT results back to 2D wavefunction
    void unflatten_wavefunction() {
        Kokkos::parallel_for("unflatten_wavefunction", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                d_psi(i, j) = d_psi_flat(i * Ny + j);
            }
        );
    }

    // Perform forward FFT (position to momentum space)
    void forwardFFT() {
        // First flatten the 2D wavefunction to 1D
        flatten_wavefunction();
        
        // Get raw pointer to the data for cuFFT
        void* d_ptr = d_psi_flat.data();
        
        // Call our cuFFT wrapper for the forward transform
        forward_fft_2d(d_ptr, Nx, Ny);
        
        // Unflatten back to 2D
        unflatten_wavefunction();
    }

    // Perform backward FFT (momentum to position space)
    void backwardFFT() {
        // First flatten the 2D wavefunction to 1D
        flatten_wavefunction();
        
        // Get raw pointer to the data for cuFFT
        void* d_ptr = d_psi_flat.data();
        
        // Call our cuFFT wrapper for the backward transform
        backward_fft_2d(d_ptr, Nx, Ny);
        
        // Unflatten back to 2D and normalize
        Kokkos::parallel_for("normalize_and_unflatten", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                int idx = i * Ny + j;
                d_psi(i, j) = d_psi_flat(idx) / static_cast<double>(N);
            }
        );
    }

    // Initialize ground state using imaginary time propagation
    void initializeGroundState() {
        // Start with a 2D Gaussian wavefunction
        double sigma = 1.0;
        
        Kokkos::parallel_for("init_gaussian", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                double xi = d_x(i);
                double yj = d_y(j);
                d_psi(i, j) = complex_type(std::exp(-(xi*xi + yj*yj)/(2.0*sigma*sigma)), 0.0);
            }
        );
        
        // Calculate initial norm
        double norm = calculateNorm();
        
        // Normalize initial state
        Kokkos::parallel_for("normalize_initial", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                d_psi(i, j) /= std::sqrt(norm);
            }
        );
        
        // Imaginary time propagation
        double imag_dt = 0.05;
        std::cout << "Starting 2D ground state calculation..." << std::endl;
        
        for (int step = 0; step < 1000; ++step) {
            // Half step potential (REAL potential V only)
            Kokkos::parallel_for("imag_time_pot1", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    d_psi(i, j) *= std::exp(-d_V(i, j) * imag_dt / 2.0);
                }
            );
            
            // Kinetic step
            forwardFFT();
            
            Kokkos::parallel_for("imag_time_kinetic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    double kx;
                    if (i <= Nx/2) kx = 2.0 * PI * i / (Nx * dx);
                    else kx = 2.0 * PI * (i - Nx) / (Nx * dx);
                    
                    double ky;
                    if (j <= Ny/2) ky = 2.0 * PI * j / (Ny * dy);
                    else ky = 2.0 * PI * (j - Ny) / (Ny * dy);
                    
                    double kinetic_factor = (kx*kx + ky*ky) / 2.0;
                    d_psi(i, j) *= std::exp(-kinetic_factor * imag_dt);
                }
            );
            
            backwardFFT();
            
            // Half step potential (REAL potential V only)
            Kokkos::parallel_for("imag_time_pot2", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
                KOKKOS_LAMBDA(const int i, const int j) {
                    d_psi(i, j) *= std::exp(-d_V(i, j) * imag_dt / 2.0);
                }
            );
            
            // Normalize
            norm = calculateNorm();
            double norm_sqrt = std::sqrt(norm);
            
            if (norm_sqrt > 1e-15) {
                Kokkos::parallel_for("normalize_step", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
                    KOKKOS_LAMBDA(const int i, const int j) {
                        d_psi(i, j) /= norm_sqrt;
                    }
                );
            } else {
                std::cerr << "Warning: Wavefunction norm close to zero during imaginary time prop." << std::endl;
            }
            
            // Occasionally calculate energy
            if (step % 100 == 0 || step == 999) {
                double energy = calculateEnergy();
                std::cout << "Step " << step << ", Energy = " << energy
                          << " a.u. (" << energy * 27.2114 << " eV)" << std::endl;
            }
        }
        
        double final_energy = calculateEnergy();
        std::cout << "2D Ground state energy: " << final_energy
                  << " a.u. (" << final_energy * 27.2114 << " eV)" << std::endl;
    }
    
    // Calculate current norm of wavefunction
    double calculateNorm() {
        // Use Kokkos reduction to calculate norm
        double norm = 0.0;
        Kokkos::parallel_reduce("calculate_norm", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j, double& partial_sum) {
                partial_sum += Kokkos::real(d_psi(i, j) * Kokkos::conj(d_psi(i, j)));
            }, norm
        );
        
        return norm * dx * dy; // Include volume element
    }
    
    // Calculate total energy
    double calculateEnergy() {
        // Calculate potential energy
        double potential_energy = 0.0;
        Kokkos::parallel_reduce("potential_energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j, double& partial_sum) {
                partial_sum += Kokkos::real(Kokkos::conj(d_psi(i, j)) * d_V(i, j) * d_psi(i, j));
            }, potential_energy
        );
        potential_energy *= dx * dy;
        
        // Create a copy of psi for kinetic energy calculation
        complex_view_2d d_psi_copy("psi_copy", Nx, Ny);
        Kokkos::deep_copy(d_psi_copy, d_psi);
        
        // Transform to k-space
        Kokkos::parallel_for("flatten_for_kinetic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                d_psi_flat(i * Ny + j) = d_psi_copy(i, j);
            }
        );
        
        // Get raw pointer for cuFFT
        void* d_ptr = d_psi_flat.data();
        forward_fft_2d(d_ptr, Nx, Ny);
        
        Kokkos::parallel_for("unflatten_for_kinetic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                d_psi_copy(i, j) = d_psi_flat(i * Ny + j);
            }
        );
        
        // Apply kinetic operator in k-space
        Kokkos::parallel_for("apply_kinetic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                double kx;
                if (i <= Nx/2) kx = 2.0 * PI * i / (Nx * dx);
                else kx = 2.0 * PI * (i - Nx) / (Nx * dx);
                
                double ky;
                if (j <= Ny/2) ky = 2.0 * PI * j / (Ny * dy);
                else ky = 2.0 * PI * (j - Ny) / (Ny * dy);
                
                double kinetic_factor = (kx*kx + ky*ky) / 2.0;
                d_psi_copy(i, j) *= kinetic_factor;
            }
        );
        
        // Transform back to position space
        Kokkos::parallel_for("flatten_for_kinetic_back", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                d_psi_flat(i * Ny + j) = d_psi_copy(i, j);
            }
        );
        
        backward_fft_2d(d_ptr, Nx, Ny);
        
        Kokkos::parallel_for("unflatten_for_kinetic_back", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                d_psi_copy(i, j) = d_psi_flat(i * Ny + j) / N; // Include normalization
            }
        );
        
        // Calculate kinetic energy
        double kinetic_energy = 0.0;
        Kokkos::parallel_reduce("kinetic_energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j, double& partial_sum) {
                partial_sum += Kokkos::real(Kokkos::conj(d_psi(i, j)) * d_psi_copy(i, j));
            }, kinetic_energy
        );
        kinetic_energy *= dx * dy;
        
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
    
    // Propagate wavefunction for one time step
    void propagateStep(double t) {
        // Calculate field values
        double E_t = electricField(t);
        double E_t_plus_dt = electricField(t + dt);
        
        // First half step potential + laser + CAP
        Kokkos::parallel_for("propagate_pot1", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                double xi = d_x(i);
                double laser_term = xi * E_t;
                double V_real_t = d_V(i, j) + laser_term;
                
                // Calculate phase factor
                double phase = -V_real_t * dt / 2.0;
                complex_type potential_phase_factor = complex_type(std::cos(phase), std::sin(phase));
                
                // Calculate CAP damping
                double cap_damping_factor = std::exp(-d_W_cap(i, j) * dt / 2.0);
                
                // Apply both effects
                d_psi(i, j) = d_psi(i, j) * potential_phase_factor * cap_damping_factor;
            }
        );
        
        // Full step kinetic
        forwardFFT();
        
        Kokkos::parallel_for("propagate_kinetic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                double kx;
                if (i <= Nx/2) kx = 2.0 * PI * i / (Nx * dx);
                else kx = 2.0 * PI * (i - Nx) / (Nx * dx);
                
                double ky;
                if (j <= Ny/2) ky = 2.0 * PI * j / (Ny * dy);
                else ky = 2.0 * PI * (j - Ny) / (Ny * dy);
                
                double kinetic_factor = (kx*kx + ky*ky) / 2.0;
                
                // Calculate phase factor
                double phase = -kinetic_factor * dt;
                complex_type kinetic_phase_factor = complex_type(std::cos(phase), std::sin(phase));
                
                // Apply phase factor
                d_psi(i, j) = d_psi(i, j) * kinetic_phase_factor;
            }
        );
        
        backwardFFT();
        
        // Second half step potential + laser + CAP
        Kokkos::parallel_for("propagate_pot2", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j) {
                double xi = d_x(i);
                double laser_term = xi * E_t_plus_dt;
                double V_real_t_plus_dt = d_V(i, j) + laser_term;
                
                // Calculate phase factor
                double phase = -V_real_t_plus_dt * dt / 2.0;
                complex_type potential_phase_factor = complex_type(std::cos(phase), std::sin(phase));
                
                // Calculate CAP damping
                double cap_damping_factor = std::exp(-d_W_cap(i, j) * dt / 2.0);
                
                // Apply both effects
                d_psi(i, j) = d_psi(i, j) * potential_phase_factor * cap_damping_factor;
            }
        );
    }
    
    // Calculate ionization probability
    double calculateIonization(double bound_radius_sq) {
        double bound_prob = 0.0;
        
        Kokkos::parallel_reduce("calculate_bound_prob", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {Nx, Ny}),
            KOKKOS_LAMBDA(const int i, const int j, double& partial_sum) {
                double xi = d_x(i);
                double yj = d_y(j);
                
                if ((xi*xi + yj*yj) < bound_radius_sq) {
                    partial_sum += Kokkos::real(d_psi(i, j) * Kokkos::conj(d_psi(i, j))) * dx * dy;
                }
            }, bound_prob
        );
        
        return 1.0 - bound_prob;
    }
    
    // Run simulation
    void run(int timesteps, double bound_radius) {
        std::ofstream outFile("ionization_2d_cap_kokkos.dat");
        std::ofstream densityFile("density_evolution_2d_cap_kokkos.dat");
        
        double t = 0.0;
        double initial_energy = calculateEnergy();
        std::cout << "Initial 2D energy (H_physical): " << initial_energy << " a.u. ("
                  << initial_energy * 27.2114 << " eV)" << std::endl;
        
        double initial_norm = calculateNorm();
        std::cout << "Initial Norm: " << initial_norm << std::endl;
        
        double bound_radius_sq = bound_radius * bound_radius;
        int density_save_interval = 500;  // Save density less often
        int density_downsample = 4;       // Save only every Nth point
        
        // Get host copy of initial state for output
        Kokkos::deep_copy(h_psi, d_psi);
        
        // Setup timer
        Kokkos::Timer timer;
        
        for (int step = 0; step < timesteps; ++step) {
            propagateStep(t);
            
            // Calculate observables periodically
            if (step % 10 == 0) {
                // Calculate current norm
                double current_norm = calculateNorm();
                
                // Calculate ionization
                double ion_prob_spatial = calculateIonization(bound_radius_sq);
                
                // Ionization as norm loss
                double ion_prob_norm_loss = initial_norm - current_norm;
                
                outFile << t << "\t"
                        << ion_prob_spatial << "\t"    // Prob outside R
                        << ion_prob_norm_loss << "\t"  // Prob absorbed
                        << electricField(t) << "\t"
                        << current_norm << std::endl;
                
                // Save density snapshot (infrequently and downsampled)
                if (step % density_save_interval == 0) {
                    double elapsed = timer.seconds();
                    std::cout << "Step " << step << ", Time: " << t << " a.u."
                              << ", Ion(space): " << ion_prob_spatial
                              << ", Norm: " << current_norm 
                              << ", Wall time: " << elapsed << " s"
                              << ", Steps/sec: " << step/elapsed << std::endl;
                    
                    // Copy current state to host for output
                    Kokkos::deep_copy(h_psi, d_psi);
                    
                    densityFile << "# Time = " << t << std::endl;
                    for (int i = 0; i < Nx; i += density_downsample) {
                        for (int j = 0; j < Ny; j += density_downsample) {
                            densityFile << h_x(i) << "\t" << h_y(j) << "\t"
                                       << Kokkos::real(h_psi(i, j) * Kokkos::conj(h_psi(i, j))) << std::endl;
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
        
        double final_norm = calculateNorm();
        std::cout << "Final Norm: " << final_norm << std::endl;
        std::cout << "Total simulation time: " << timer.seconds() << " seconds" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    
    {
        // Simulation Parameters (Adjust Carefully!)
        int Nx = 512;
        int Ny = 512;
        double dx = 0.4;
        double dy = 0.4;
        double dt = 0.03;
        double a = 0.79837;
        
        // Laser parameters
        double E0 = 0.142622;
        double omega = 0.057;
        double cycles = 3.0;
        double phase = 0.0;
        
        // Analysis parameters
        double bound_radius = 15.0;
        int timesteps = 10000;
        
        // CAP Parameters
        double grid_radius_x = (Nx / 2.0) * dx;
        double grid_radius_y = (Ny / 2.0) * dy;
        double cap_start_x = grid_radius_x * 0.75;
        double cap_start_y = grid_radius_y * 0.75;
        double cap_eta = 0.05;
        
        std::cout << "2D Hydrogen in laser field simulation with CAP (Kokkos GPU version)" << std::endl;
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
        
        // Print Kokkos configuration
        std::cout << "\nKokkos Configuration:" << std::endl;
        std::cout << "  Execution Space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;
        std::cout << "  Memory Space: " << typeid(Kokkos::DefaultExecutionSpace::memory_space).name() << std::endl;
        
        try {
            HydrogenSimulator2D simulator(Nx, Ny, dx, dy, dt, a,
                                         E0, omega, cycles, phase,
                                         cap_start_x, cap_start_y, cap_eta);
            
            simulator.run(timesteps, bound_radius);
            
            std::cout << "Simulation complete." << std::endl;
            std::cout << "Data saved to ionization_2d_cap_kokkos.dat and density_evolution_2d_cap_kokkos.dat" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            Kokkos::finalize();
            return 1;
        } catch (...) {
            std::cerr << "An unknown error occurred." << std::endl;
            Kokkos::finalize();
            return 1;
        }
    }
    
    // Finalize Kokkos
    Kokkos::finalize();
    
    return 0;
}
