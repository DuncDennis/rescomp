import numpy as np

def _kuramoto_sivashinsky(dimensions, system_size, dt, time_steps, starting_point):
    """ This function simulates the Kuramoto–Sivashinsky PDE

    Even though it doesn't use the RK4 algorithm, it is bundled with the other
    simulation functions in simulate_trajectory() for consistency.

    Reference for the numerical integration:
    "fourth order time stepping for stiff pde-kassam trefethen 2005" at
    https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf

    Python implementation at: https://github.com/E-Renshaw/kuramoto-sivashinsky

    Args:
        dimensions (int): nr. of dimensions of the system grid
        system_size (int): physical size of the system
        dt (float): time step size
        time_steps (int): nr. of time steps to simulate
        starting_point (np.ndarray): starting point for the simulation of shape
            (dimensions, )

    Returns:
        (np.ndarray): simulated trajectory of shape (time_steps, dimensions)

    """
    n = dimensions  # No. of grid points in real space (and hence dimensionality of the output)
    size = system_size  # system size

    # Define initial conditions and Fourier Transform them
    if starting_point is None:
        # Use the starting point from the Kassam_2005 paper
        x = size * np.transpose(np.conj(np.arange(1, n + 1))) / n
        u = np.cos(2 * np.pi * x / size) * (1 + np.sin(2 * np.pi * x / size))
    else:
        # x = starting_point
        u = starting_point

    v = np.fft.fft(u)

    h = dt  # time step
    nmax = time_steps  # No. of time steps to simulate

    # Wave numbers
    k = np.transpose(
        np.conj(np.concatenate((np.arange(0, n / 2), np.array([0]),
                                np.arange(-n / 2 + 1, 0))))) * 2 * np.pi / size

    # Just copied from the paper, it works
    L = k ** 2 - k ** 4
    E = np.exp(h * L)
    E_2 = np.exp(h * L / 2)
    M = int(np.ceil(size/(2 * np.pi)))
    # M = int(size/(2 * np.pi))
    # M = (size * np.pi) // 2
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], n, axis=0)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(
        np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3,
                axis=1))
    f2 = h * np.real(
        np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = h * np.real(
        np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3,
                axis=1))

    uu = [np.array(
        u)]  # List of Real space solutions, later converted to a np.array

    g = -0.5j * k  # TODO: Meaning?

    # See paper for details
    for n in range(1, nmax + 1):
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v)) ** 2)
        a = E_2 * v + Q * Nv
        Na = g * np.fft.fft(np.real(np.fft.ifft(a)) ** 2)
        b = E_2 * v + Q * Na
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b)) ** 2)
        c = E_2 * a + Q * (2 * Nb - Nv)
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c)) ** 2)

        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        u = np.real(np.fft.ifft(v))
        uu.append(np.array(u))

    uu = np.array(uu)
    # print("PDE simulation finished")

    return uu


simulation_time_steps = 1000
time_steps=simulation_time_steps
dimensions=40
system_size=22
dt = 0.5
starting_point = None
sim_data = _kuramoto_sivashinsky(dimensions, system_size, dt, time_steps, starting_point)

############# ADDED ################
print(sim_data)

indics_of_nan = np.where(np.isnan(sim_data))
print(indics_of_nan)

# for i in sim_data:
#     print(i[0])
