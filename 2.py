import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from Euler import euler
from EulerKromer import euler_kromer
from EMP import emp
from KD import kd
from RK4 import rk4

# ============================================================
# Настройки
# ============================================================
ct = 2000.0             # общее время интегрирования (одинаковое для всех шагов)
IC_RANGE = (0.0, 1.0)   # диапазон случайных начальных условий
N_IC = 5                # число начальных условий; объём усредняется по ним
SEED = 12345            # одинаковые старты у всех методов и RK4
params = (1.0, 1.0, 1.0)  # (a1, a2, a3) фиксированы

# Значения шага: линейная сетка от 0.001 до 0.1
h_values = np.linspace(0.001, 0.1, 100)

# Фиксированная высота окна по Y для каждого метода.
# None -> подбирается автоматически (95-й перцентиль данных * 1.3)
y_lims = {
    "Euler": None,
    "Euler-Cromer": None,
    "EMP": None,
    "VSKD": None,
}
AUTO_PERCENTILE = 95
AUTO_MARGIN = 1.3

save_folder = "t2_phase_volume_vs_h_plots"
os.makedirs(save_folder, exist_ok=True)


# ============================================================
# Система: Sprott Case B
# ============================================================
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


# ============================================================
# Расчёт фазового объёма для всех шагов (параллельно по шагам)
# ============================================================
@njit(parallel=True)
def phase_volume_vs_h(method, f, h_vals, u_init, ct, a1, a2, a3):
    n = h_vals.shape[0]
    n_ic = u_init.shape[0]
    volumes = np.full(n, np.nan)

    for k in prange(n):
        h = h_vals[k]
        total = 0.0
        cnt = 0

        for j in range(n_ic):
            t_arr, s = method(f, u_init[j].copy(), 0.0, ct, h, a1, a2, a3)

            # Проверка: не разошлось ли решение
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
                # V = (x_max-x_min)(y_max-y_min)(z_max-z_min), без transient
                vx = np.max(s[:, 0]) - np.min(s[:, 0])
                vy = np.max(s[:, 1]) - np.min(s[:, 1])
                vz = np.max(s[:, 2]) - np.min(s[:, 2])
                total += vx * vy * vz
                cnt += 1

        # Если разошлось хотя бы одно ИУ -> точка не строится (NaN)
        if cnt == n_ic:
            volumes[k] = total / n_ic

    return volumes


def run_scan(method, u_init):
    return phase_volume_vs_h(
        method, f, h_values, u_init, ct, params[0], params[1], params[2]
    )


# ============================================================
# График
# ============================================================
def get_ylim(method_name, vol_m, vol_rk4):
    manual = y_lims.get(method_name)
    if manual is not None:
        return (0, manual)

    data = np.concatenate([vol_m, vol_rk4])
    data = data[np.isfinite(data)]
    if data.size == 0:
        return (0, 1)
    top = np.percentile(data, AUTO_PERCENTILE) * AUTO_MARGIN
    return (0, top if top > 0 else 1.0)


def plot_phase_volume(method_name, vol_m, vol_rk4):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    plt.plot(h_values, vol_m, color="blue", linewidth=1.0, label=method_name)
    plt.plot(h_values, vol_rk4, color="red", linewidth=1.0, alpha=0.6,
             label="RK4")
    plt.xlabel("Шаг h")
    plt.ylabel("Фазовый объём")
    plt.title(f"Sprott Case B: Фазовый объём от шага h, "
              f"{method_name} и RK4")
    plt.legend(loc="best")

    plt.xlim(min(h_values), max(h_values))
    plt.ylim(*get_ylim(method_name, vol_m, vol_rk4))

    filepath = os.path.join(
        save_folder, f"PhaseVol_vs_h_Case_B_{method_name}.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    u_init = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=(N_IC, 3))

    print("RK4...")
    vol_rk4 = run_scan(rk4_jit, u_init)

    for method_name, method in methods:
        print(f"{method_name}...")
        vol_m = run_scan(method, u_init)
        plot_phase_volume(method_name, vol_m, vol_rk4)

    print("\nПостроение всех графиков фазового объёма от шага завершено.")