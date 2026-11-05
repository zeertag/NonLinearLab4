import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from KD import kd

# ============================================================
# Настройки
# ============================================================
h = 0.001
ct = 2000.0             # общее время интегрирования
IC_RANGE = (0.0, 1.0)   # диапазон случайных начальных условий
N_IC = 5                # число начальных условий; объём усредняется по ним
SEED = 12345            # фиксируем начальные условия для воспроизводимости
params = (1.0, 1.0, 1.0)  # (a1, a2, a3) фиксированы

# Диапазоны коэффициента симметрии S
S_ranges = [
    ("0_1", np.linspace(0.0, 1.0, 1000)),
    ("m100_100", np.linspace(-100.0, 100.0, 2000)),
]

# Фиксированная высота окна по Y для каждого диапазона.
# None -> подбирается автоматически (95-й перцентиль данных * 1.3)
y_lims = {
    "0_1": None,
    "m100_100": None,
}
AUTO_PERCENTILE = 95
AUTO_MARGIN = 1.3

save_folder = "t3_phase_volume_vs_S_plots"
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


kd_jit = njit(kd)


# ------------------------------------------------------------
# ВАЖНО: адаптер метода КД.
# Здесь предполагается, что КД принимает коэффициент симметрии S
# последним аргументом: kd(f, u0, t0, t1, h, a1, a2, a3, S).
# Если в вашей реализации сигнатура другая, поправьте только эту функцию.
# ------------------------------------------------------------
@njit(cache=True)
def kd_with_S(f, u0, t0, t1, h, a1, a2, a3, S):
    return kd_jit(f, u0, t0, t1, h, a1, a2, a3, S)


# ============================================================
# Расчёт фазового объёма
# ============================================================
@njit(cache=True)
def volume_of(s):
    """Фазовый объём траектории или NaN, если решение разошлось."""
    for i in range(s.shape[0]):
        if (not np.isfinite(s[i, 0]) or not np.isfinite(s[i, 1])
                or not np.isfinite(s[i, 2])):
            return np.nan
        if np.max(np.abs(s[i])) > 1e6:
            return np.nan

    vx = np.max(s[:, 0]) - np.min(s[:, 0])
    vy = np.max(s[:, 1]) - np.min(s[:, 1])
    vz = np.max(s[:, 2]) - np.min(s[:, 2])
    return vx * vy * vz


@njit(parallel=True)
def phase_volume_vs_S(method, f, S_vals, u_init, h, ct, a1, a2, a3):
    """КД при разных S (параллельно по значениям S)."""
    n = S_vals.shape[0]
    n_ic = u_init.shape[0]
    volumes = np.full(n, np.nan)

    for k in prange(n):
        total = 0.0
        cnt = 0
        for j in range(n_ic):
            t_arr, s = method(f, u_init[j].copy(), 0.0, ct, h,
                              a1, a2, a3, S_vals[k])
            v = volume_of(s)
            if np.isfinite(v):
                total += v
                cnt += 1

        # Если разошлось хотя бы одно ИУ -> точка не строится (NaN)
        if cnt == n_ic:
            volumes[k] = total / n_ic

    return volumes


# ============================================================
# График
# ============================================================
def get_ylim(key, vol_kd):
    manual = y_lims.get(key)
    if manual is not None:
        return (0, manual)

    data = vol_kd[np.isfinite(vol_kd)]
    if data.size == 0:
        return (0, 1)
    top = np.percentile(data, AUTO_PERCENTILE) * AUTO_MARGIN
    return (0, top if top > 0 else 1.0)


def plot_phase_volume(key, S_vals, vol_kd):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)
    plt.plot(S_vals, vol_kd, color="blue", linewidth=1.0, label="VSKD")
    plt.xlabel("Коэффициент симметрии S")
    plt.ylabel("Фазовый объём")
    plt.title("Sprott Case B: Фазовый объём от коэффициента симметрии S, "
              "VSKD")
    plt.legend(loc="best")

    plt.xlim(min(S_vals), max(S_vals))
    plt.ylim(*get_ylim(key, vol_kd))

    filepath = os.path.join(
        save_folder, f"PhaseVol_vs_S_Case_B_VSKD_{key}.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    u_init = rng.uniform(IC_RANGE[0], IC_RANGE[1], size=(N_IC, 3))

    for key, S_vals in S_ranges:
        print(f"VSKD, диапазон S: {key}...")
        vol_kd = phase_volume_vs_S(
            kd_with_S, f, S_vals, u_init, h, ct,
            params[0], params[1], params[2]
        )
        plot_phase_volume(key, S_vals, vol_kd)

    print("\nПостроение всех графиков фазового объёма от S завершено.")