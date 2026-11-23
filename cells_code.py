"""Код ячеек ноутбука lab3.ipynb. Ключи соответствуют порядку в build_notebook.py."""

CODE = {}

CODE["imports"] = r'''
%matplotlib inline
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist, pdist

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score,
    rand_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    fowlkes_mallows_score, homogeneity_completeness_v_measure,
)
from sklearn.metrics.cluster import pair_confusion_matrix, contingency_matrix

warnings.filterwarnings("ignore")
SEED = 42
pd.set_option("display.precision", 3)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100
'''

CODE["load"] = r'''
df = pd.read_csv("data/WineQT.csv")
print("Размерность исходной таблицы:", df.shape)
display(df.head())

FEATURES = [c for c in df.columns if c not in ("quality", "Id")]
X_raw = df[FEATURES].copy()
quality = df["quality"].copy()

info = pd.DataFrame({
    "тип": X_raw.dtypes.astype(str),
    "пропуски": X_raw.isna().sum(),
    "уникальных": X_raw.nunique(),
})
display(info)
print("Пропусков всего:", int(df.isna().sum().sum()))
print("Полных дубликатов по физико-химическим признакам:", int(X_raw.duplicated().sum()))
print("Распределение оценок качества:", quality.value_counts().sort_index().to_dict())
'''

CODE["describe"] = r'''
desc = X_raw.describe().T
desc["CV, %"] = 100 * desc["std"] / desc["mean"]
desc["асимметрия"] = X_raw.skew()
desc["эксцесс"] = X_raw.kurt()
desc["p (D'Agostino)"] = [stats.normaltest(X_raw[c]).pvalue for c in FEATURES]
display(desc[["mean", "std", "min", "25%", "50%", "75%", "max", "CV, %", "асимметрия", "эксцесс", "p (D'Agostino)"]])
'''

CODE["hist"] = r'''
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, FEATURES):
    sns.histplot(X_raw[col], bins=35, kde=True, ax=ax, color="#4C72B0", stat="density")
    mu, sd = X_raw[col].mean(), X_raw[col].std()
    xs = np.linspace(X_raw[col].min(), X_raw[col].max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), "r--", lw=1.2, label="N(μ, σ)")
    ax.set_title(f"{col}\nasym = {X_raw[col].skew():.2f}", fontsize=10)
    ax.set_xlabel("")
axes.flat[0].legend(fontsize=8)
axes.flat[-1].axis("off")
fig.suptitle("Распределения исходных признаков (красная линия – нормальное распределение с теми же μ и σ)", y=1.0)
plt.tight_layout()
plt.show()
'''

CODE["informativeness"] = r'''
corr = X_raw.corr(method="spearman")
fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"width_ratios": [1.25, 1]})
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[0],
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
axes[0].set_title("Корреляции Спирмена между признаками")

mi = pd.Series(mutual_info_classif(X_raw, quality, random_state=SEED), index=FEATURES)
signif = pd.DataFrame({
    "CV, %": desc["CV, %"],
    "max |r| с другими": corr.abs().where(~np.eye(len(FEATURES), dtype=bool)).max(),
    "MI с quality": mi,
}).sort_values("MI с quality", ascending=False)
signif["MI с quality"].plot.barh(ax=axes[1], color="#55A868")
axes[1].invert_yaxis()
axes[1].set_title("Взаимная информация признаков с оценкой quality")
axes[1].set_xlabel("MI, нат")
plt.tight_layout()
plt.show()
display(signif)
'''

CODE["outliers_raw"] = r'''
q1, q3 = X_raw.quantile(0.25), X_raw.quantile(0.75)
iqr = q3 - q1
iqr_mask = (X_raw < q1 - 1.5 * iqr) | (X_raw > q3 + 1.5 * iqr)
z_mask = pd.DataFrame(np.abs(stats.zscore(X_raw)) > 3, columns=FEATURES)
outl = pd.DataFrame({"IQR (1.5)": iqr_mask.sum(), "|z| > 3": z_mask.sum()})
display(outl.T)
print("Строк хотя бы с одним выбросом по IQR:", int(iqr_mask.any(axis=1).sum()),
      f"({iqr_mask.any(axis=1).mean():.1%})")
print("Строк хотя бы с одним выбросом по |z| > 3:", int(z_mask.any(axis=1).sum()),
      f"({z_mask.any(axis=1).mean():.1%})")

Xs_tmp = StandardScaler().fit_transform(X_raw)
fig, ax = plt.subplots(figsize=(14, 4.5))
sns.boxplot(data=pd.DataFrame(Xs_tmp, columns=FEATURES), ax=ax, color="#8DA0CB", fliersize=2)
ax.set_title("Диаграммы размаха исходных признаков (z-оценки)")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.show()

# как выбросы влияют на K-means, если их не обрабатывать
for k in [4, 5, 6]:
    sizes = np.bincount(KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xs_tmp))
    print(f"K-means на сырых данных, k = {k}: размеры кластеров {sorted(sizes.tolist())}")
'''

CODE["transform"] = r'''
SKEWED = ["residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "sulphates"]
X_log = X_raw.copy()
X_log[SKEWED] = np.log1p(X_log[SKEWED])

z_log = np.abs(stats.zscore(X_log))
keep = (z_log <= 3).all(axis=1)
print("Удалено наблюдений с |z| > 3 после логарифмирования:", int((~keep).sum()),
      f"({(~keep).mean():.1%})")

X_clean_raw = X_raw[keep].reset_index(drop=True)     # исходные единицы – для интерпретации центров
X_clean = X_log[keep].reset_index(drop=True)         # пространство кластеризации
y_quality = quality[keep].reset_index(drop=True)
print("Размер очищенной выборки:", X_clean.shape)

skew_cmp = pd.DataFrame({
    "асимметрия: исходные": X_raw.skew(),
    "после log1p и очистки": X_clean.skew(),
})
display(skew_cmp.T)

fig, axes = plt.subplots(1, 5, figsize=(18, 3.4))
for ax, col in zip(axes, SKEWED):
    sns.histplot(X_clean[col], bins=30, kde=True, ax=ax, color="#C44E52")
    ax.set_title(f"log1p({col})\nasym = {X_clean[col].skew():.2f}", fontsize=10)
    ax.set_xlabel("")
plt.tight_layout()
plt.show()
'''

CODE["hopkins"] = r'''
def hopkins(X, frac=0.1, seed=SEED):
    """Статистика Хопкинса H = Σu / (Σu + Σw): u – расстояния от равномерно
    распределённых точек до ближайшего объекта выборки, w – от объектов выборки
    до их ближайшего соседа. H ≈ 0.5 – случайные данные, H → 1 – выраженная кластерность."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    n, m = len(X), int(frac * len(X))
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    w = nn.kneighbors(X[rng.choice(n, m, replace=False)], 2)[0][:, 1]
    U = rng.uniform(X.min(0), X.max(0), (m, X.shape[1]))
    u = nn.kneighbors(U, 1)[0][:, 0]
    return u.sum() / (u.sum() + w.sum())

Xh = StandardScaler().fit_transform(X_clean)
H = [hopkins(Xh, seed=s) for s in range(20)]
rng = np.random.default_rng(SEED)
H_unif = [hopkins(rng.uniform(Xh.min(0), Xh.max(0), Xh.shape), seed=s) for s in range(20)]
print(f"Хопкинс, данные:            {np.mean(H):.3f} ± {np.std(H):.3f}")
print(f"Хопкинс, равномерный шум:   {np.mean(H_unif):.3f} ± {np.std(H_unif):.3f}")
'''

CODE["scaling"] = r'''
scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}
rows = []
for name, sc in scalers.items():
    Z = sc.fit_transform(X_clean)
    lab = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(Z)
    spread = Z.std(axis=0)
    rows.append({
        "метод": name,
        "разброс σ признаков (min–max)": f"{spread.min():.2f}–{spread.max():.2f}",
        "силуэт (k=2)": silhouette_score(Z, lab),
        "CH (k=2)": calinski_harabasz_score(Z, lab),
        "размеры кластеров": sorted(np.bincount(lab).tolist()),
    })
display(pd.DataFrame(rows).set_index("метод"))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)
display(X_scaled_df.describe().T[["mean", "std", "min", "max"]].round(3))
'''

CODE["pairplot"] = r'''
PAIR = ["fixed acidity", "volatile acidity", "citric acid", "density", "pH", "sulphates", "alcohol",
        "total sulfur dioxide"]
g = sns.pairplot(X_scaled_df[PAIR], corner=True, diag_kind="kde",
                 plot_kws={"s": 8, "alpha": 0.45, "edgecolor": "none"}, height=1.85)
g.figure.suptitle("Матрица диаграмм рассеивания (стандартизованные признаки)", y=1.01)
plt.show()

pca_full = PCA(random_state=SEED).fit(X_scaled)
ev = pca_full.explained_variance_ratio_
X_pca = pca_full.transform(X_scaled)[:, :2]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 12), ev, color="#4C72B0", label="доля дисперсии")
axes[0].plot(range(1, 12), np.cumsum(ev), "o-", color="#C44E52", label="накопленная доля")
axes[0].set_xlabel("номер главной компоненты"); axes[0].set_xticks(range(1, 12)); axes[0].legend()
axes[0].set_title("Объяснённая дисперсия PCA")
axes[1].hexbin(X_pca[:, 0], X_pca[:, 1], gridsize=35, cmap="Blues", mincnt=1)
axes[1].set_xlabel(f"PC1 ({ev[0]:.1%})"); axes[1].set_ylabel(f"PC2 ({ev[1]:.1%})")
axes[1].set_title("Плотность объектов в плоскости PC1–PC2")
plt.tight_layout()
plt.show()
print("Накопленная доля дисперсии первых 2/3/5 компонент:", np.cumsum(ev)[[1, 2, 4]].round(3))
loadings = pd.DataFrame(pca_full.components_[:3].T, index=FEATURES, columns=["PC1", "PC2", "PC3"])
display(loadings.round(3))
'''

CODE["elbow"] = r'''
K_RANGE = range(2, 11)
rows = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_scaled)
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X_scaled)
    rows.append({
        "k": k, "WCSS (inertia)": km.inertia_,
        "силуэт KM": silhouette_score(X_scaled, km.labels_),
        "CH KM": calinski_harabasz_score(X_scaled, km.labels_),
        "DB KM": davies_bouldin_score(X_scaled, km.labels_),
        "силуэт Ward": silhouette_score(X_scaled, ag.labels_),
        "CH Ward": calinski_harabasz_score(X_scaled, ag.labels_),
        "DB Ward": davies_bouldin_score(X_scaled, ag.labels_),
    })
k_tab = pd.DataFrame(rows).set_index("k")
# второй разностный критерий для «локтя»
k_tab["Δ² WCSS"] = k_tab["WCSS (inertia)"].diff().diff().shift(-1)
display(k_tab)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
axes[0].plot(k_tab.index, k_tab["WCSS (inertia)"], "o-")
axes[0].set_title("Метод локтя (K-means)"); axes[0].set_xlabel("k"); axes[0].set_ylabel("WCSS")
axes[1].plot(k_tab.index, k_tab["силуэт KM"], "o-", label="K-means")
axes[1].plot(k_tab.index, k_tab["силуэт Ward"], "s--", label="Ward")
axes[1].set_title("Средний коэффициент силуэта"); axes[1].set_xlabel("k"); axes[1].legend()
axes[2].plot(k_tab.index, k_tab["CH KM"], "o-", label="CH, K-means")
axes[2].plot(k_tab.index, k_tab["CH Ward"], "s--", label="CH, Ward")
ax2 = axes[2].twinx()
ax2.plot(k_tab.index, k_tab["DB KM"], "^:", color="gray", label="DB, K-means")
axes[2].set_title("Calinski–Harabasz (слева) и Davies–Bouldin (справа)"); axes[2].set_xlabel("k")
axes[2].legend(loc="upper center"); ax2.legend(loc="center right"); ax2.grid(False)
plt.tight_layout()
plt.show()
'''

CODE["silhouette_profiles"] = r'''
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True)
for ax, k in zip(axes, [2, 3, 4]):
    lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X_scaled)
    s = silhouette_samples(X_scaled, lab)
    y0 = 10
    for c in range(k):
        sc = np.sort(s[lab == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(sc)), 0, sc, alpha=0.8, color=sns.color_palette("Set2")[c])
        ax.text(-0.08, y0 + len(sc) / 2, str(c))
        y0 += len(sc) + 10
    ax.axvline(s.mean(), color="red", ls="--")
    ax.set_title(f"k = {k}: средний силуэт {s.mean():.3f}, доля s<0: {(s < 0).mean():.1%}")
    ax.set_yticks([]); ax.set_xlabel("коэффициент силуэта")
plt.tight_layout()
plt.show()
'''

CODE["dendrogram"] = r'''
Z_ward = linkage(X_scaled, method="ward")
cut2 = (Z_ward[-1, 2] + Z_ward[-2, 2]) / 2        # уровень разреза, дающий два кластера
fig, axes = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw={"width_ratios": [2.2, 1]})
dendrogram(Z_ward, truncate_mode="lastp", p=40, show_leaf_counts=True, leaf_rotation=90,
           leaf_font_size=8, ax=axes[0], color_threshold=cut2)
axes[0].axhline(cut2, color="red", ls="--", lw=1, label="разрез на k = 2")
axes[0].legend()
axes[0].set_title("Дендрограмма (метод Уорда, последние 40 слияний)")
axes[0].set_ylabel("расстояние слияния")
last = Z_ward[-12:, 2][::-1]
axes[1].plot(range(2, 14), last, "o-")
axes[1].set_xticks(range(2, 14))
axes[1].set_title("Высоты последних 12 слияний")
axes[1].set_xlabel("k – число кластеров при разрезе чуть ниже слияния")
plt.tight_layout()
plt.show()
gaps = pd.Series(-np.diff(last), index=[f"k = {k}" for k in range(2, 13)])
print("Ширина интервала высот, в котором разрез даёт k кластеров:")
print(gaps.round(2).head(6).to_string())
'''

CODE["fit_models"] = r'''
K_OPT = 2
kmeans = KMeans(n_clusters=K_OPT, n_init=10, random_state=SEED).fit(X_scaled)
labels_km = kmeans.labels_
ward = AgglomerativeClustering(n_clusters=K_OPT, linkage="ward").fit(X_scaled)
labels_hc = ward.labels_

# согласуем номера: кластер 1 – группа с более высокой фиксированной кислотностью
def align(lab):
    m = pd.Series(X_clean["fixed acidity"]).groupby(lab).mean()
    return lab if m[1] > m[0] else 1 - lab
labels_km, labels_hc = align(labels_km), align(labels_hc)

print("Размеры кластеров K-means:", np.bincount(labels_km))
print("Размеры кластеров Ward:   ", np.bincount(labels_hc))
print("Число итераций K-means:", kmeans.n_iter_, " | инерция:", round(kmeans.inertia_, 1))
cross = pd.crosstab(pd.Series(labels_km, name="K-means"), pd.Series(labels_hc, name="Ward"))
display(cross)
print(f"Совпадение меток двух методов: {(labels_km == labels_hc).mean():.1%}, "
      f"ARI между методами: {adjusted_rand_score(labels_km, labels_hc):.3f}")
'''

CODE["metrics"] = r'''
def jaccard_pairs(a, b):
    (tn, fp), (fn, tp) = pair_confusion_matrix(a, b)
    return tp / (tp + fp + fn)

def purity(y, lab):
    cm = contingency_matrix(y, lab)
    return cm.max(axis=0).sum() / cm.sum()

y_bin = (y_quality >= 6).astype(int).values   # «хорошее» (≥ 6) / «обычное» (≤ 5)

def all_metrics(lab):
    h, c, v = homogeneity_completeness_v_measure(y_bin, lab)
    return {
        "Силуэт ↑": silhouette_score(X_scaled, lab),
        "Calinski–Harabasz ↑": calinski_harabasz_score(X_scaled, lab),
        "Davies–Bouldin ↓": davies_bouldin_score(X_scaled, lab),
        "Rand index ↑": rand_score(y_bin, lab),
        "Adjusted Rand ↑": adjusted_rand_score(y_bin, lab),
        "Jaccard (пары) ↑": jaccard_pairs(y_bin, lab),
        "Fowlkes–Mallows ↑": fowlkes_mallows_score(y_bin, lab),
        "NMI ↑": normalized_mutual_info_score(y_bin, lab),
        "Homogeneity ↑": h, "Completeness ↑": c,
        "Purity ↑": purity(y_bin, lab),
    }

metrics = pd.DataFrame({"K-means (k=2)": all_metrics(labels_km), "Ward (k=2)": all_metrics(labels_hc)})
# опорные значения: случайное разбиение тех же размеров
rng = np.random.default_rng(SEED)
metrics["случайные метки"] = pd.DataFrame(
    [all_metrics(rng.permutation(labels_km)) for _ in range(20)]).mean()
display(metrics.round(3))

for name, lab in [("K-means", labels_km), ("Ward", labels_hc)]:
    ct = pd.crosstab(pd.Series(lab, name=f"кластер {name}"), y_quality.rename("quality"))
    ct["доля quality ≥ 6"] = (ct[[6, 7, 8]].sum(axis=1) / ct.sum(axis=1)).round(3)
    display(ct)
'''

CODE["distances"] = r'''
def cluster_geometry(X, lab):
    ks = np.unique(lab)
    centers = np.vstack([X[lab == c].mean(0) for c in ks])
    rows = []
    for c in ks:
        P = X[lab == c]
        d_c = np.linalg.norm(P - centers[c], axis=1)
        rows.append({
            "кластер": c, "n": len(P),
            "ср. расст. до центра": d_c.mean(),
            "RMS-радиус": np.sqrt((d_c ** 2).mean()),
            "ср. попарное расст.": pdist(P).mean(),
            "диаметр": pdist(P).max(),
            "WCSS": (d_c ** 2).sum(),
        })
    geo = pd.DataFrame(rows).set_index("кластер")
    D = cdist(X[lab == 0], X[lab == 1])
    between = {
        "между центрами": np.linalg.norm(centers[0] - centers[1]),
        "ближайшие точки (single)": D.min(),
        "среднее (average)": D.mean(),
        "дальние точки (complete)": D.max(),
    }
    return geo, pd.Series(between), centers

for name, lab in [("K-means", labels_km), ("Ward", labels_hc)]:
    geo, between, centers = cluster_geometry(X_scaled, lab)
    print(f"=== {name} ===")
    display(geo.round(3))
    print("Межкластерные расстояния:", between.round(3).to_dict())
    print("Отношение межцентрового расстояния к среднему радиусу:",
          round(between["между центрами"] / geo["ср. расст. до центра"].mean(), 3))
'''

CODE["centers"] = r'''
cent_scaled = pd.DataFrame(
    {f"KM {c}": X_scaled[labels_km == c].mean(0) for c in [0, 1]} |
    {f"Ward {c}": X_scaled[labels_hc == c].mean(0) for c in [0, 1]}, index=FEATURES)

cent_raw = pd.DataFrame(
    {f"KM {c}": X_clean_raw[labels_km == c].median() for c in [0, 1]} |
    {f"Ward {c}": X_clean_raw[labels_hc == c].median() for c in [0, 1]} |
    {"вся выборка": X_clean_raw.median()})
cent_raw.loc["quality (среднее)"] = [y_quality[labels_km == 0].mean(), y_quality[labels_km == 1].mean(),
                                     y_quality[labels_hc == 0].mean(), y_quality[labels_hc == 1].mean(),
                                     y_quality.mean()]
print("Центры кластеров в стандартизованном пространстве (средние z-оценки):")
display(cent_scaled.round(2))
print("Медианы признаков по кластерам в исходных единицах:")
display(cent_raw.round(3))

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(cent_scaled.T, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Профили центров кластеров (средние z-оценки признаков)")
plt.tight_layout()
plt.show()
'''

CODE["param_k"] = r'''
# 1) влияние k: внутренние и внешние метрики (внешняя – по трём уровням качества)
y3 = pd.cut(y_quality, [0, 5, 6, 10], labels=[0, 1, 2]).astype(int).values   # ≤5 / 6 / ≥7
rows = []
for k in range(2, 11):
    lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X_scaled)
    sizes = np.bincount(lab)
    rows.append({"k": k, "силуэт": silhouette_score(X_scaled, lab),
                 "CH": calinski_harabasz_score(X_scaled, lab), "DB": davies_bouldin_score(X_scaled, lab),
                 "ARI (3 уровня)": adjusted_rand_score(y3, lab),
                 "NMI (3 уровня)": normalized_mutual_info_score(y3, lab),
                 "мин. кластер": sizes.min(), "макс. кластер": sizes.max()})
param_k = pd.DataFrame(rows).set_index("k")
display(param_k.round(3))
'''

CODE["param_init"] = r'''
# 2) влияние способа инициализации и n_init на итоговую инерцию и устойчивость
rows = []
for k in [2, 3, 4, 6]:
    for init in ["k-means++", "random"]:
        for n_init in [1, 10]:
            runs = [KMeans(n_clusters=k, init=init, n_init=n_init, random_state=s).fit(X_scaled)
                    for s in range(30)]
            inert = np.array([r.inertia_ for r in runs])
            ref = runs[int(inert.argmin())].labels_
            ari = np.array([adjusted_rand_score(ref, r.labels_) for r in runs])
            rows.append({"k": k, "init": init, "n_init": n_init,
                         "инерция min": inert.min(), "инерция mean": inert.mean(),
                         "доля запусков с лучшей инерцией": np.isclose(inert, inert.min(), rtol=1e-6).mean(),
                         "ARI с лучшим решением (mean)": ari.mean()})
init_tab = pd.DataFrame(rows).set_index(["k", "init", "n_init"])
display(init_tab.round(3))
'''

CODE["param_stability"] = r'''
# 3) устойчивость к выборке: бутстреп 80 % объектов, сравнение с решением на полной выборке
def bootstrap_stability(make_model, k, n_boot=30):
    full = make_model(k).fit_predict(X_scaled)
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(n_boot):
        idx = rng.choice(len(X_scaled), int(0.8 * len(X_scaled)), replace=False)
        sub = make_model(k).fit_predict(X_scaled[idx])
        out.append(adjusted_rand_score(full[idx], sub))
    return np.mean(out), np.std(out)

rows = []
for k in [2, 3, 4, 5]:
    m_km, s_km = bootstrap_stability(lambda k: KMeans(n_clusters=k, n_init=10, random_state=SEED), k)
    m_hc, s_hc = bootstrap_stability(lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward"), k)
    rows.append({"k": k, "K-means ARI": f"{m_km:.3f} ± {s_km:.3f}", "Ward ARI": f"{m_hc:.3f} ± {s_hc:.3f}"})
display(pd.DataFrame(rows).set_index("k"))
'''

CODE["param_linkage"] = r'''
# 4) влияние метода связи в иерархической кластеризации (k = 2 и 3)
rows = []
for link in ["ward", "complete", "average", "single"]:
    for k in [2, 3]:
        lab = AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(X_scaled)
        rows.append({"linkage": link, "k": k, "силуэт": silhouette_score(X_scaled, lab),
                     "CH": calinski_harabasz_score(X_scaled, lab),
                     "размеры": sorted(np.bincount(lab).tolist(), reverse=True),
                     "ARI с K-means": adjusted_rand_score(
                         KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X_scaled), lab)})
display(pd.DataFrame(rows).set_index(["linkage", "k"]).round(3))
'''

CODE["viz_pca"] = r'''
pal = {0: "#4C72B0", 1: "#DD8452"}
cent_km_pca = pca_full.transform(np.vstack([X_scaled[labels_km == c].mean(0) for c in [0, 1]]))[:, :2]
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharex=True, sharey=True)
for ax, lab, title in [(axes[0], labels_km, "K-means, k = 2"), (axes[1], labels_hc, "Ward, k = 2")]:
    for c in [0, 1]:
        ax.scatter(X_pca[lab == c, 0], X_pca[lab == c, 1], s=10, alpha=0.6, color=pal[c], label=f"кластер {c}")
    ax.set_title(title); ax.legend()
axes[0].scatter(cent_km_pca[:, 0], cent_km_pca[:, 1], marker="X", s=250, c="black", label="центры")
diff = labels_km != labels_hc
axes[2].scatter(X_pca[~diff, 0], X_pca[~diff, 1], s=8, color="lightgray", label="метки совпали")
axes[2].scatter(X_pca[diff, 0], X_pca[diff, 1], s=14, color="crimson", label=f"метки различаются ({diff.sum()})")
axes[2].set_title("Расхождение методов"); axes[2].legend()
for ax in axes:
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
axes[0].set_ylabel(f"PC2 ({ev[1]:.1%})")
plt.tight_layout()
plt.show()
'''

CODE["viz_pairs"] = r'''
VIS = ["fixed acidity", "citric acid", "volatile acidity", "pH", "density", "alcohol"]
plot_df = X_clean_raw[VIS].copy()
plot_df["кластер K-means"] = labels_km
g = sns.pairplot(plot_df, hue="кластер K-means", palette=pal, corner=True, diag_kind="kde",
                 plot_kws={"s": 9, "alpha": 0.5, "edgecolor": "none"}, height=2.0)
g.figure.suptitle("Кластеры K-means в пространстве исходных признаков", y=1.01)
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, col in zip(axes, ["fixed acidity", "citric acid", "volatile acidity", "alcohol"]):
    sns.boxplot(x=labels_km, y=X_clean_raw[col], hue=labels_km, palette=pal, ax=ax, legend=False)
    ax.set_xlabel("кластер K-means"); ax.set_title(col)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
pd.crosstab(labels_km, y_quality, normalize="index").plot.bar(stacked=True, ax=ax, colormap="RdYlGn", rot=0)
ax.set_xlabel("кластер K-means"); ax.set_ylabel("доля образцов"); ax.legend(title="quality", bbox_to_anchor=(1, 1))
ax.set_title("Распределение экспертных оценок внутри кластеров")
plt.tight_layout()
plt.show()
'''

CODE["viz_k3"] = r'''
# Дополнительно: более детальное разбиение k = 3 (второй локальный кандидат по CH и дендрограмме)
labels_km3 = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit_predict(X_scaled)
order = pd.Series(X_clean_raw["fixed acidity"]).groupby(labels_km3).mean().sort_values().index
remap = {old: new for new, old in enumerate(order)}
labels_km3 = np.vectorize(remap.get)(labels_km3)
prof3 = X_clean_raw.groupby(labels_km3).median().T
prof3.loc["quality (среднее)"] = y_quality.groupby(labels_km3).mean()
prof3.loc["n"] = np.bincount(labels_km3)
prof3.loc["доля quality ≥ 6"] = (y_quality >= 6).groupby(labels_km3).mean()
display(prof3.round(3))
fig, ax = plt.subplots(figsize=(7, 5.5))
for c, col in zip(range(3), ["#4C72B0", "#DD8452", "#55A868"]):
    ax.scatter(X_pca[labels_km3 == c, 0], X_pca[labels_km3 == c, 1], s=10, alpha=0.6, color=col, label=f"кластер {c}")
ax.set_xlabel(f"PC1 ({ev[0]:.1%})"); ax.set_ylabel(f"PC2 ({ev[1]:.1%})"); ax.legend()
ax.set_title("K-means, k = 3 в плоскости PCA")
plt.tight_layout()
plt.show()
'''

CODE["summary"] = r'''
summary = pd.DataFrame({
    "K-means (k=2)": [*metrics["K-means (k=2)"][["Силуэт ↑", "Calinski–Harabasz ↑", "Davies–Bouldin ↓",
                                                 "Adjusted Rand ↑", "Jaccard (пары) ↑", "NMI ↑"]],
                      kmeans.inertia_],
    "Ward (k=2)": [*metrics["Ward (k=2)"][["Силуэт ↑", "Calinski–Harabasz ↑", "Davies–Bouldin ↓",
                                           "Adjusted Rand ↑", "Jaccard (пары) ↑", "NMI ↑"]],
                   sum(((X_scaled[labels_hc == c] - X_scaled[labels_hc == c].mean(0)) ** 2).sum() for c in [0, 1])],
}, index=["Силуэт", "Calinski–Harabasz", "Davies–Bouldin", "ARI (quality ≥ 6)", "Jaccard (quality ≥ 6)",
          "NMI (quality ≥ 6)", "WCSS"])
display(summary.round(3))
'''
