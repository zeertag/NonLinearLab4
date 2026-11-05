import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from Euler import euler
from EulerKromer import euler_kromer
from EMP import emp
from KD import kd
from RK4 import rk4


h = 0.001
ct = 2000.0
IC_RANGE = (0.0, 1.0)
SEED = 12345
start_params = (1.0, 1.0, 1.0)

a1_values = np.linspace(0, 5, 100)
a2_values = np.linspace(0, 5, 100)
a3_values = np.linspace(0, 5, 100)

y_lims = {
    "a1": 3000,
    "a2": 3000,
    "a3": 7000,
}



save_folder = "t1_phase_volume_plots"
os.makedirs(save_folder, exist_ok=True)

@njit(cache=True)
def f(u, a1, a2, a3):
    x = u[0]
    y = u[1]
    z = u[2]
    return np.array([
        a1 * y * z,
        x - a2 * y,
        a3 - x * x
    ])

methods = [
    ("Euler", njit(euler)),
    ("Euler-Cromer", njit(euler_kromer)),
    ("EMP", njit(emp)),
    ("VSKD", njit(kd)),
]
rk4_jit = njit(rk4)

parameters = [
    ("a1", a1_values),
    ("a2", a2_values),
    ("a3", a3_values),
]

@njit(parallel=True)
def phase_volume_scan(method, f, values, which, u_init, h, ct, a1_0, a2_0, a3_0):
    n = values.shape[0]
    volumes = np.full(n, np.nan)

    for k in prange(n):
        a1, a2, a3 = a1_0, a2_0, a3_0
        if which == 0:
            a1 = values[k]
        elif which == 1:
            a2 = values[k]
        else:
            a3 = values[k]

        t_arr, s = method(f, u_init[k].copy(), 0.0, ct, h, a1, a2, a3)

        ok = True
        for i in range(s.shape[0]):
            if (not np.isfinite(s[i, 0]) or not np.isfinite(s[i, 1])
                    or not np.isfinite(s[i, 2])):
                ok = False
                break
            if np.max(np.abs(s[i])) > 1e6:
                ok = False
                break

        if ok:
            x_min = np.min(s[:, 0])
            x_max = np.max(s[:, 0])
            y_min = np.min(s[:, 1])
            y_max = np.max(s[:, 1])
            z_min = np.min(s[:, 2])
            z_max = np.max(s[:, 2])
            volumes[k] = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    return volumes


def run_scan(method, which, values, u_init):
    return phase_volume_scan(
        method, f, values, which, u_init, h, ct,
        start_params[0], start_params[1], start_params[2]
    )

def plot_phase_volume(method_name, parameter_name, params, vol_m, vol_rk4):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    plt.plot(params, vol_m, color="blue", linewidth=1.0, label=method_name)
    plt.plot(params, vol_rk4, color="red", linewidth=1.0, alpha=0.6,
             label="RK4")
    plt.xlabel(parameter_name)
    plt.ylabel("Фазовый объём")
    plt.title(f"Sprott Case B: Фазовый объём от {parameter_name}, "
              f"{method_name} и RK4")
    plt.legend(loc="best")

    plt.xlim(min(params), max(params))
    plt.ylim(0, y_lims[parameter_name])

    filepath = os.path.join(
        save_folder,
        f"PhaseVol_Case_B_{parameter_name}_{method_name}.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    for which, (parameter_name, params) in enumerate(parameters):
        n = len(params)
        u_init = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=(n, 3))

        print(f"RK4, параметр {parameter_name}...")
        vol_rk4 = run_scan(rk4_jit, which, params, u_init)

        for method_name, method in methods:
            print(f"{method_name}, параметр {parameter_name}...")
            vol_m = run_scan(method, which, params, u_init)
            plot_phase_volume(method_name, parameter_name, params, vol_m, vol_rk4)

    print("\nПостроение всех графиков фазового объёма завершено.")
