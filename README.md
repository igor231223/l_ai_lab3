# Лабораторная работа №3. Кластеризация

Отчёт – ноутбук [`lab3.ipynb`](lab3.ipynb) (выполнен, все графики и таблицы сохранены в нём).

Датасет: [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) – `data/WineQT.csv`.
Методы: K-means и агломеративная иерархическая кластеризация (метод Уорда), k = 2.

## Состав

| Файл | Назначение |
|---|---|
| `lab3.ipynb` | отчёт: разделы 1–8 по заданию |
| `data/WineQT.csv` | датасет |
| `build_notebook.py` | сборка ноутбука из исходников |
| `cells_code.py` | код вычислительных ячеек |
| `md_text.py` | текст отчёта (Markdown-ячейки) |

## Запуск

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab3.ipynb
```

## Основные результаты (k = 2)

| Метрика | K-means | Ward |
|---|---|---|
| Силуэт | 0.204 | 0.198 |
| Calinski–Harabasz | 247.3 | 219.3 |
| Davies–Bouldin | 1.909 | 1.966 |
| ARI (quality ≥ 6) | 0.042 | 0.068 |
| Устойчивость (ARI на подвыборках) | 0.935 | 0.744 |
