"""Сборка ноутбука lab3.ipynb из кода (cells_code.py) и текста (md_text.py).

Запуск:
    .venv/bin/python build_notebook.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace lab3.ipynb
"""
from pathlib import Path

import nbformat as nbf

from cells_code import CODE
from md_text import MD

ROOT = Path(__file__).parent

# порядок ячеек: ("md", ключ) или ("code", ключ)
ORDER = [
    ("md", "intro"),
    ("md", "dataset"),
    ("code", "imports"),
    ("code", "load"),
    ("md", "load_res"),
    ("code", "describe"),
    ("md", "describe_res"),
    ("code", "hist"),
    ("md", "hist_res"),
    ("code", "informativeness"),
    ("md", "informativeness_res"),
    ("code", "outliers_raw"),
    ("md", "outliers_res"),
    ("md", "preproc"),
    ("code", "transform"),
    ("md", "transform_res"),
    ("code", "hopkins"),
    ("md", "hopkins_res"),
    ("md", "scaling_intro"),
    ("code", "scaling"),
    ("md", "scaling_res"),
    ("code", "pairplot"),
    ("md", "pairplot_res"),
    ("md", "work_intro"),
    ("code", "elbow"),
    ("md", "elbow_res"),
    ("code", "silhouette_profiles"),
    ("md", "silhouette_res"),
    ("code", "dendrogram"),
    ("md", "dendrogram_res"),
    ("md", "fit_intro"),
    ("code", "fit_models"),
    ("md", "fit_res"),
    ("md", "metrics_intro"),
    ("code", "metrics"),
    ("md", "metrics_res"),
    ("code", "distances"),
    ("md", "distances_res"),
    ("code", "centers"),
    ("md", "centers_res"),
    ("md", "param_intro"),
    ("code", "param_k"),
    ("md", "param_k_res"),
    ("code", "param_init"),
    ("md", "param_init_res"),
    ("code", "param_stability"),
    ("md", "param_stability_res"),
    ("code", "param_linkage"),
    ("md", "param_linkage_res"),
    ("md", "viz_intro"),
    ("code", "viz_pca"),
    ("md", "viz_pca_res"),
    ("code", "viz_pairs"),
    ("md", "viz_pairs_res"),
    ("code", "viz_k3"),
    ("md", "viz_k3_res"),
    ("code", "summary"),
    ("md", "summary_res"),
    ("md", "conclusion"),
    ("md", "sources"),
]


def build():
    cells = []
    for kind, key in ORDER:
        if kind == "md":
            cells.append(nbf.v4.new_markdown_cell(MD.get(key, f"### TODO: {key}").strip("\n")))
        else:
            cells.append(nbf.v4.new_code_cell(CODE[key].strip("\n")))

    # Приложение: полный листинг кода всех ячеек
    listing = "\n\n".join(CODE[key].strip("\n") for kind, key in ORDER if kind == "code")
    cells.append(nbf.v4.new_markdown_cell(
        MD["appendix"].strip("\n") + "\n\n```python\n" + listing + "\n```"))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.write(nb, ROOT / "lab3.ipynb")
    print(f"lab3.ipynb: {len(cells)} ячеек")


if __name__ == "__main__":
    build()
