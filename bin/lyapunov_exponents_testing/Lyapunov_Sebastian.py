import numpy as np
import matplotlib.pyplot as plt
import contextlib  # for temp_seed


def _kuramoto_sivashinsky_custom(dimensions, system_size, dt, time_steps, starting_point,
                                 precision=None, fft_type=None, M=64, **kwargs):
    """ This function simulates the Kuramoto–Sivashinsky PDE with custom precision and fft backend"""

    if precision is None:
        change_precision = False
    elif precision == 128:
        change_precision = True
        f_dtype = 'float128'
        c_dtype = 'complex256'
    elif precision == 64:
        change_precision = True
        f_dtype = 'float64'
        c_dtype = 'complex128'
    elif precision == 32:
        change_precision = True
        f_dtype = 'float32'
        c_dtype = 'complex64'
    elif precision == 16:
        change_precision = True
        f_dtype = 'float16'
        c_dtype = 'complex32'
    else:
        raise Exception("specified precision not recognized")

    if fft_type is None or fft_type == 'numpy':
        custom_fft = np.fft.fft
        custom_ifft = np.fft.ifft
    elif fft_type == 'scipy':
        import scipy
        try:
            import scipy.fft
            custom_fft = scipy.fft.fft
            custom_ifft = scipy.fft.ifft
        except ModuleNotFoundError:
            # Depricated, but needed for older versions of scipy
            import scipy.fftpack
            custom_fft = scipy.fftpack.fft
            custom_ifft = scipy.fftpack.ifft
    elif fft_type == 'pyfftw_np':
        import pyfftw
        custom_fft = pyfftw.interfaces.numpy_fft.fft
        custom_ifft = pyfftw.interfaces.numpy_fft.ifft
    elif fft_type == 'pyfftw_sc':
        import pyfftw
        custom_fft = pyfftw.interfaces.scipy_fft.fft
        custom_ifft = pyfftw.interfaces.scipy_fft.ifft
    elif fft_type == 'pyfftw_fftw':
        import pyfftw
        a = pyfftw.empty_aligned(dimensions, dtype='complex128')
        b = pyfftw.empty_aligned(dimensions, dtype='complex128')
        c = pyfftw.empty_aligned(dimensions, dtype='complex128')
        fft_object = pyfftw.FFTW(a, b)
        ifft_object = pyfftw.FFTW(b, c, direction='FFTW_BACKWARD')
        custom_fft = fft_object
        custom_ifft = ifft_object
    else:
        raise Exception('fft_type not recognized')

    # Rename variables to the names used in the paper
    n = dimensions  # No. of grid points in real space (and hence dimensionality of the output)
    size = system_size  # system size
    h = dt  # time step
    nmax = time_steps  # No. of time steps to simulate

    # Define initial conditions and Fourier Transform them
    if starting_point is None:
        # Use the starting point from the Kassam_2005 paper
        x = size * np.transpose(np.conj(np.arange(1, n + 1))) / n
        u = np.cos(2 * np.pi * x / size) * (1 + np.sin(2 * np.pi * x / size))
    else:
        u = starting_point
    if change_precision: u = u.astype(np.single)
    v = custom_fft(u)

    # Wave numbers
    k = np.transpose(np.conj(np.concatenate((
        np.arange(0, n / 2), np.array([0]), np.arange(-n / 2 + 1, 0))
    ))) * 2 * np.pi / size
    if change_precision: k = k.astype(f_dtype)

    L = k ** 2 - k ** 4
    E = np.exp(h * L)
    if change_precision: E = E.astype(f_dtype)
    E_2 = np.exp(h * L / 2)
    if change_precision: E_2 = E_2.astype(f_dtype)
    # M = 64
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    if change_precision: r = r.astype(c_dtype)
    LR = h * np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], n, axis=0)
    if change_precision: LR = LR.astype(c_dtype)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    if change_precision: Q = Q.astype(c_dtype)
    f1 = h * np.real(
        np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3,
                axis=1))
    if change_precision: f1 = f1.astype(c_dtype)
    f2 = h * np.real(
        np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    if change_precision: f2 = f2.astype(c_dtype)
    f3 = h * np.real(
        np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3,
                axis=1))
    if change_precision: f3 = f3.astype(c_dtype)

    # List of Real space solutions, later converted to a np.array
    uu = [np.array(u)]

    g = -0.5j * k
    if change_precision: g = g.astype(c_dtype)

    # See paper for details
    for n in range(1, nmax):
        Nv = g * custom_fft(np.real(custom_ifft(v)) ** 2)
        if change_precision: Nv = Nv.astype(c_dtype)
        a = E_2 * v + Q * Nv
        if change_precision: a = a.astype(c_dtype)
        Na = g * custom_fft(np.real(custom_ifft(a)) ** 2)
        if change_precision: Na = Na.astype(c_dtype)
        b = E_2 * v + Q * Na
        if change_precision: b = b.astype(c_dtype)
        Nb = g * custom_fft(np.real(custom_ifft(b)) ** 2)
        if change_precision: Nb = Nb.astype(c_dtype)
        c = E_2 * a + Q * (2 * Nb - Nv)
        if change_precision: c = c.astype(c_dtype)
        Nc = g * custom_fft(np.real(custom_ifft(c)) ** 2)
        if change_precision: Nc = Nc.astype(c_dtype)

        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        # v = E * v
        # v += Nv * f1
        # v += 2 * (Na + Nb) * f2
        # v += Nc * f3
        if change_precision: v = v.astype(c_dtype)
        u = np.real(custom_ifft(v))
        uu.append(np.array(u))

    uu = np.array(uu)
    # print("PDE simulation finished")

    return uu


@contextlib.contextmanager
def temp_seed(seed):
    """
    from https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    Use like:
    with temp_seed(5):
        <do_smth_that_uses_np.random>

    """

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def plot_log_divergence(log_div_list, dt=1.0, fit=True, t_min=None, t_max=None, figsize=(9, 4), ax=None):
    time_steps = log_div_list.size
    t_list = np.arange(time_steps) * dt

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    round_digs = 5

    if fit:
        if t_min is None:
            t_min = 0
        if t_max is None:
            t_max = t_list[-1]

        x_fit, y_fit, coefs = _linear_fit(log_div_list, dt=dt, t_min=t_min, t_max=t_max)

        ax.plot(x_fit, y_fit,
                label=f"Sloap = {np.round(coefs[0], round_digs)}, Intersect = {np.round(coefs[1], round_digs)}",
                linestyle="--", c="k")

    ax.plot(t_list, log_div_list)
    ax.grid()
    ax.legend()
    ax.set_xlabel("time")
    ax.set_ylabel(r"avg. log distance")


def _linear_fit(y, dt, t_min=None, t_max=None):
    """
    y: list of numbers to be fitted
    dt: time_step
    t_min, t_max: fitting range

    returns x_fit and y_fit (for plotting) and coef = (sloap, intersept)
    """
    if t_min is None:
        i_min = 0
    else:
        i_min = int(t_min / dt)

    if t_max is None:
        i_max = y.size - 1
    else:
        i_max = int(t_max / dt)

    y = y[i_min: i_max+1]
    x_fit = np.arange(i_min, i_max+1)*dt
    coef = np.polyfit(x_fit, y, 1)
    poly1d_fn = np.poly1d(coef)
    y_fit = poly1d_fn(x_fit)
    return x_fit, y_fit, coef


def simple_largest_lyapunov(f, starting_point, n_parts=5, t_part=1.0, dt=1.0, n_disc=10, seed=None, eps=1e-10):
    """
    Like QR decomposition algo, but only for largest LE.
    f: iterator -> x_(i+1) = f(x_(i))
    starting_point: np.ndarray

    n_parts: Number of re-normalization steps
    t_part: Time between re-normalization steps
    n_disc: Number of re-normalization steps to discard in beginning
    eps: initial perturbation length
    """
    def f_steps(x, steps):
        for i in range(steps):
            x = f(x)
        return x

    state_dim = starting_point.size

    t_part_timesteps = int(t_part / dt)

    x = starting_point
    if seed is not None:
        with temp_seed(seed):
            perturbation = np.random.randn(state_dim)
    else:
        perturbation = np.random.randn(state_dim)

    le_avg_list = np.zeros(n_parts)
    perturbation *= eps/np.linalg.norm(perturbation)

    for i_n in range(n_disc):
        if (i_n+1) % 10 == 0:
            print(f"discard: {i_n+1}/{n_disc}", end="\r")
        x_perturbed_initial = x + perturbation
        x = f_steps(x, t_part_timesteps)
        x_perturbed_evolved = f_steps(x_perturbed_initial, t_part_timesteps)
        perturbation_evolved = x_perturbed_evolved - x
        perturbed_length = np.linalg.norm(perturbation_evolved)
        perturbation = eps / perturbed_length * perturbation_evolved

    print("\n")
    for i_n in range(n_parts):
        if (i_n+1) % 10 == 0:
            print(f"{i_n + 1}/{n_parts}", end="\r")
        x_perturbed_initial = x + perturbation
        x = f_steps(x, t_part_timesteps)
        x_perturbed_evolved = f_steps(x_perturbed_initial, t_part_timesteps)
        perturbation_evolved = x_perturbed_evolved - x

        perturbed_length = np.linalg.norm(perturbation_evolved)
        local_le = np.log(perturbed_length/eps)/(dt*t_part_timesteps)
        if i_n == 0:
            le_avg = local_le
        else:
            le_avg = (le_avg*(i_n) + local_le)/(i_n+1)
        le_avg_list[i_n] = le_avg
        perturbation = eps / perturbed_length * perturbation_evolved

    return le_avg_list


def simple_largest_lyapunov_traj_div(f, starting_point, n_parts=5, t_part=1.0, dt=1.0, n_disc=10, seed=None, eps=1e-10):
    """
    Similar to above function. Saves the trajectory divergence during the time_window (t_part) and averages along the
    trajectory.
    """
    def f_steps(x, steps):
        for i in range(steps):
            x = f(x)
        return x

    state_dim = starting_point.size

    t_part_timesteps = int(t_part / dt)

    x = starting_point
    if seed is not None:
        with temp_seed(seed):
            perturbation = np.random.randn(state_dim)
    else:
        perturbation = np.random.randn(state_dim)

    perturbation *= eps/np.linalg.norm(perturbation)

    for i_n in range(n_disc):
        if (i_n+1) % 10 == 0:
            print(f"discard: {i_n+1}/{n_disc}", end="\r")
        x_perturbed = x + perturbation
        x = f_steps(x, t_part_timesteps)
        x_perturbed = f_steps(x_perturbed, t_part_timesteps)
        perturbation = (x_perturbed - x)*(eps/np.linalg.norm((x_perturbed - x)))

    distances = np.zeros((t_part_timesteps, n_parts))
    print("\n")
    for i_n in range(n_parts):
        if (i_n+1) % 10 == 0:
            print(f"{i_n + 1}/{n_parts}", end="\r")
        x_perturbed = x + perturbation
        for i_t in range(t_part_timesteps):
            x = f(x)
            x_perturbed = f(x_perturbed)
            distance = np.linalg.norm(x_perturbed - x)
            distances[i_t, i_n] = distance
        perturbation = (x_perturbed - x)*(eps/distance)

    mean_log_distances = np.mean(np.log(distances), axis=-1)
    return mean_log_distances


if __name__ == "__main__":
    system_size = 36
    dimensions = 54
    precision = None
    fft_type = "scipy"

    def data_creation_function(time_steps, dt, starting_point=None):
        return _kuramoto_sivashinsky_custom(dimensions, system_size, dt, time_steps,
                                                                starting_point,
                                                                precision=precision, fft_type=fft_type)


    dt = 0.25
    f = lambda x: data_creation_function(2, dt, starting_point=x)[-1]

    seed = 5
    np.random.seed(seed)
    starting_point = np.random.randn(dimensions)

    # first experiment: "QR-type" algorithm:
    n_parts = 10 ** 4
    n_disc = 10 ** 4
    t_part = 0.5
    eps = 1e-6

    lle_convergence = simple_largest_lyapunov(f, starting_point, n_parts=n_parts,
                                                               t_part=t_part, dt=dt, n_disc=n_disc, eps=eps)

    plt.figure(figsize=(9, 4))
    plt.plot(lle_convergence)
    plt.axhline(lle_convergence[-1], linestyle="--", color="r")
    plt.title(f"Largest LE: {lle_convergence[-1]}")
    plt.savefig("lle_convergence")

    # second experiment: trajectory divergence and linear fit:
    n_parts = 10 ** 3
    n_disc = 10 ** 3
    t_part = 3
    eps = 1e-6
    mean_log_distances = simple_largest_lyapunov_traj_div(f, starting_point, n_parts=n_parts,
                                                                  t_part=t_part, dt=dt, n_disc=n_disc, eps=eps)
    plot_log_divergence(mean_log_distances, dt=dt, t_min=None, t_max=None)
    plt.savefig("log_divergence")
