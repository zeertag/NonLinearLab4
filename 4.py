import os
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from Euler import euler
from EulerKromer import euler_kromer
from EMP import emp
from KD import kd
from RK4 import rk4

res = 100
t_run = 1000.0
u0 = np.array([0.1, 0.1, 0.1], dtype=float)
params = (1.0, 1.0, 1.0)
MAX_PEAKS = 2000

h_values = np.array([0.01, 0.005, 0.001])
colors = ["tab:green", "tab:blue", "tab:purple"]


variables = [
    (0, "x", "X", None),
]

save_folder = "t4_continuation_plots"
for _, _, folder, _ in variables:
    os.makedirs(os.path.join(save_folder, folder), exist_ok=True)


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
    ("RK4", njit(rk4)),
]


@njit(cache=True)
def find_peaks(x, out):
    cnt = 0
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            if cnt < out.shape[0]:
                out[cnt] = x[i]
                cnt += 1
    return cnt


@njit(parallel=True)
def continuation_scan(method, f, h_vals, u_start, t1, n_res,
                      a1, a2, a3, max_peaks):

    nh = h_vals.shape[0]
    buf = np.zeros((nh, 3, n_res, max_peaks))
    counts = np.zeros((nh, 3, n_res), dtype=np.int64)

    for k in prange(nh):
        u = u_start.copy()

        for r in range(n_res):
            t_arr, sol = method(f, u.copy(), 0.0, t1, h_vals[k], a1, a2, a3)

            # решение разошлось -> дальше не считаем
            ok = True
            for i in range(sol.shape[0]):
                for j in range(3):
                    v = sol[i, j]
                    if not np.isfinite(v) or abs(v) > 1e6:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

            for c in range(3):
                counts[k, c, r] = find_peaks(sol[:, c], buf[k, c, r])

            u = sol[sol.shape[0] - 1].copy()

    return buf, counts


def run_scan(method):
    return continuation_scan(
        method, f, h_values, u0, t_run, res,
        params[0], params[1], params[2], MAX_PEAKS
    )


def collect(buf, counts, k, c):
    runs = np.concatenate(
        [np.full(cnt, r + 1) for r, cnt in enumerate(counts[k, c])]
    )
    vals = np.concatenate(
        [buf[k, c, r, :cnt] for r, cnt in enumerate(counts[k, c])]
    )
    return runs, vals


def get_ylim(manual, all_vals):
    if manual is not None:
        return manual
    if all_vals.size == 0:
        return (-1, 1)
    lo, hi = np.percentile(all_vals, [0.5, 99.5])
    pad = 0.1 * (hi - lo if hi > lo else 1.0)
    return (lo - pad, hi + pad)


def plot_continuation(method_name, idx, var_name, folder, manual_ylim,
                      buf, counts):
    plt.figure(figsize=(10, 6))
    plt.grid(True, alpha=0.3)

    dot = dict(marker=".", linestyle="none", markersize=3,
               markeredgewidth=0)

    all_vals = []
    for k, h in enumerate(h_values):
        runs, vals = collect(buf, counts, k, idx)
        if len(vals) > 0:
            plt.plot(runs, vals, color=colors[k % len(colors)],
                     alpha=0.6, label=f"h = {h}", **dot)
            all_vals.append(vals)

    all_vals = np.concatenate(all_vals) if all_vals else np.array([])

    plt.xlabel("Номер прогона")
    plt.ylabel(f"Максимумы {var_name}")
    plt.title(f"Sprott Case B: диаграмма продолжения, {method_name} "
              f"(переменная {var_name})")
    plt.legend(loc="best", markerscale=4)
    plt.xlim(0, res + 1)
    plt.ylim(*get_ylim(manual_ylim, all_vals))

    filepath = os.path.join(
        save_folder, folder, f"Continuation_Case_B_{method_name}.png"
    )
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":
    for method_name, method in methods:
        print(f"{method_name}...")
        buf, counts = run_scan(method)

        for idx, var_name, folder, ylim in variables:
            plot_continuation(method_name, idx, var_name, folder, ylim,
                              buf, counts)

    print("\nПостроение всех диаграмм продолжения завершено.")