import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from numba import njit, prange
from sklearn.cluster import KMeans

from Euler import euler
from EulerKromer import euler_kromer
from EMP import emp
from KD import kd
from RK4 import rk4

# ============================================================
# Настройки
# ============================================================

# Параметры системы (a1, a2, a3). ВАЖНО: их нужно подобрать по
# бифуркационным диаграммам так, чтобы в системе было несколько аттракторов.
params = (1.0, 1.0, 1.0)

# Сетка начальных условий: z0 фиксировано, перебираем x0 и y0
z0 = 0.1
X_RANGE = (-4.0, 4.0)
Y_RANGE = (-4.0, 4.0)
RES = 100               # разрешение сетки RES x RES

# Интегрирование
h = 0.01
t_end = 300.0
transient = 0.5         # доля траектории, отбрасываемая перед анализом пиков.
                        # Для бассейнов переход от НУ к аттрактору нужно
                        # отсечь, иначе НУ попадёт в статистику.

# Кластеризация по (средняя амплитуда пика, средний межпиковый интервал)
N_CLUSTERS = 2          # число режимов (аттракторов), которые ищем

save_folder = "t5_basins_plots"

methods = [
    ("Euler", njit(euler)),
    ("Euler-Cromer", njit(euler_kromer)),
    ("EMP", njit(emp)),
    ("VSKD", njit(kd)),
    ("RK4", njit(rk4)),
]


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


# ============================================================
# Расчёт признаков для каждого начального условия
# ============================================================
@njit(parallel=True)
def basin_scan(method, f, x0s, y0s, z0, t1, h, a1, a2, a3, transient):
    """
    Для каждой пары (x0, y0) считает по переменной x:
      amp - средняя амплитуда (среднее значение локальных максимумов x),
      per - средний межпиковый интервал.
    Если решение разошлось -> NaN. Если пиков меньше двух (нет колебаний),
    amp = среднее x на анализируемом участке, per = 0.
    """
    nx = x0s.shape[0]
    ny = y0s.shape[0]
    amp = np.full((ny, nx), np.nan)
    per = np.full((ny, nx), np.nan)

    for idx in prange(nx * ny):
        i = idx // nx   # индекс по y0
        j = idx % nx    # индекс по x0

        u0 = np.array([x0s[j], y0s[i], z0])
        t, sol = method(f, u0, 0.0, t1, h, a1, a2, a3)

        n = sol.shape[0]

        # решение разошлось?
        ok = True
        for k in range(n):
            for c in range(3):
                v = sol[k, c]
                if not np.isfinite(v) or abs(v) > 1e6:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        cut = int(n * transient)
        if cut < 1:
            cut = 1

        cnt = 0
        s = 0.0
        t_first = 0.0
        t_last = 0.0
        for k in range(cut, n - 1):
            if sol[k, 0] > sol[k - 1, 0] and sol[k, 0] > sol[k + 1, 0]:
                if cnt == 0:
                    t_first = t[k]
                t_last = t[k]
                s += sol[k, 0]
                cnt += 1

        if cnt >= 2:
            amp[i, j] = s / cnt
            per[i, j] = (t_last - t_first) / (cnt - 1)
        else:
            m = 0.0
            for k in range(cut, n):
                m += sol[k, 0]
            amp[i, j] = m / (n - cut)
            per[i, j] = 0.0

    return amp, per


# ============================================================
# Кластеризация
# ============================================================
def cluster(amp, per):
    """
    Двумерная кластеризация (KMeans) по признакам (amp, per).
    Возвращает карту меток (-1 = решение разошлось) и центры кластеров
    в исходных единицах. Кластеры нумеруются по возрастанию амплитуды.
    """
    ny, nx = amp.shape
    labels = np.full(amp.size, -1, dtype=int)

    a = amp.ravel()
    p = per.ravel()
    valid = np.isfinite(a) & np.isfinite(p)

    if valid.sum() == 0:
        return labels.reshape(ny, nx), np.zeros((0, 2))

    F = np.column_stack([a[valid], p[valid]])

    # стандартизация, чтобы признаки были сопоставимы по масштабу
    mean = F.mean(axis=0)
    std = F.std(axis=0)
    std[std == 0] = 1.0
    Fn = (F - mean) / std

    k = min(N_CLUSTERS, len(np.unique(Fn, axis=0)))
    if k < 1:
        k = 1
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Fn)
    raw = km.labels_

    # перенумерация по возрастанию средней амплитуды кластера
    centers = np.array([F[raw == c].mean(axis=0) for c in range(k)])
    order = np.argsort(centers[:, 0])
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)

    labels[valid] = remap[raw]
    return labels.reshape(ny, nx), centers[order]


# ============================================================
# Графики
# ============================================================
extent = [X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]]


def plot_map(data, title, cbar_label, filepath, cmap_name="viridis"):
    """Карта значения признака. Цветовая шкала по 1-99 перцентилям,
    выбросы не портят картину; разошедшиеся решения - серые."""
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("lightgray")

    finite = data[np.isfinite(data)]
    if finite.size > 0:
        vmin, vmax = np.percentile(finite, [1, 99])
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    plt.figure(figsize=(8, 7))
    im = plt.imshow(np.ma.masked_invalid(data), origin="lower",
                    extent=extent, aspect="auto", cmap=cmap,
                    vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, label=cbar_label)
    plt.xlabel("x0")
    plt.ylabel("y0")
    plt.title(title)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


def plot_basin(labels, centers, title, filepath):
    k = len(centers)
    palette = plt.get_cmap("tab10").colors[:max(k, 1)]
    cmap = ListedColormap(palette).copy()
    cmap.set_bad("lightgray")

    plt.figure(figsize=(8, 7))
    plt.imshow(np.ma.masked_equal(labels, -1), origin="lower",
               extent=extent, aspect="auto", cmap=cmap,
               vmin=-0.5, vmax=max(k, 1) - 0.5, interpolation="nearest")

    handles = [
        Patch(color=palette[c],
              label=f"кластер {c}: A = {centers[c, 0]:.3g}, "
                    f"T = {centers[c, 1]:.3g}")
        for c in range(k)
    ]
    if np.any(labels == -1):
        handles.append(Patch(color="lightgray", label="решение разошлось"))
    plt.legend(handles=handles, loc="upper right", fontsize=8)

    plt.xlabel("x0")
    plt.ylabel("y0")
    plt.title(title)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"График сохранен: {filepath}")


if __name__ == "__main__":
    x0s = np.linspace(X_RANGE[0], X_RANGE[1], RES)
    y0s = np.linspace(Y_RANGE[0], Y_RANGE[1], RES)

    for method_name, method in methods:
        print(f"{method_name}...")
        amp, per = basin_scan(
            method, f, x0s, y0s, z0, t_end, h,
            params[0], params[1], params[2], transient
        )
        labels, centers = cluster(amp, per)

        folder = os.path.join(save_folder, method_name)
        os.makedirs(folder, exist_ok=True)

        plot_map(
            amp,
            f"Sprott Case B, {method_name}: средняя амплитуда (z0 = {z0})",
            "Средняя амплитуда пика x",
            os.path.join(folder, f"Basin_amplitude_{method_name}.png"),
        )
        plot_map(
            per,
            f"Sprott Case B, {method_name}: средний межпиковый интервал "
            f"(z0 = {z0})",
            "Средний межпиковый интервал",
            os.path.join(folder, f"Basin_interval_{method_name}.png"),
            cmap_name="plasma",
        )
        plot_basin(
            labels, centers,
            f"Sprott Case B, {method_name}: бассейны притяжения "
            f"(z0 = {z0})",
            os.path.join(folder, f"Basin_clusters_{method_name}.png"),
        )

    print("\nПостроение всех бассейнов притяжения завершено.")