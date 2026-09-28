"""
create_notebooks.py — generates all 5 Jupyter notebooks for the Cyber Dashboard.
Run with: .venv\Scripts\python.exe create_notebooks.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent / "notebooks"
ROOT.mkdir(exist_ok=True)


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.9"},
        },
        "cells": cells,
    }


def md(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src,
        "id": "md" + str(abs(hash(src)))[:6],
    }


def code(src):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None,
        "id": "co" + str(abs(hash(src)))[:6],
    }


# ── 01 Baseline Modelling ─────────────────────────────────────────────────────

n01 = nb([
    md("# 01 · Baseline Modelling\n\n"
       "Compare a Logistic Regression baseline against Random Forest on a "
       "stratified 20 000-row sample of the CICIDS2017 dataset."),
    code(
        "import sys\n"
        'sys.path.insert(0, "..")\n'
        "import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')\n"
        "from src.data.loader import get_stratified_sample, get_numeric_columns\n"
        "from src.data.preprocessing import clean_dataframe, encode_labels, scale_features\n"
        'print("Imports OK")'
    ),
    md("## Load 20 000-row stratified sample"),
    code(
        "df = get_stratified_sample(n=20_000)\n"
        'print(f"Sample shape: {df.shape}")\n'
        'print(df["Label"].value_counts().head(10))'
    ),
    md("## Feature preparation"),
    code(
        'numeric_cols = get_numeric_columns(df)\n'
        "df_clean = clean_dataframe(df, numeric_cols)\n"
        "X = df_clean[numeric_cols]\n"
        'y_enc, enc = encode_labels(df["Label"])\n'
        "X_tr_s, X_te_s, scaler = scale_features(X.iloc[:16000], X.iloc[16000:])\n"
        "y_tr, y_te = y_enc[:16000], y_enc[16000:]\n"
        'print(f"Train: {X_tr_s.shape}  Test: {X_te_s.shape}  Classes: {enc.classes_.tolist()}")'
    ),
    md("## Baseline: Logistic Regression"),
    code(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score, f1_score\n"
        "lr = LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')\n"
        "lr.fit(X_tr_s, y_tr)\n"
        "y_pred_lr = lr.predict(X_te_s)\n"
        "acc_lr = accuracy_score(y_te, y_pred_lr)\n"
        "f1_lr  = f1_score(y_te, y_pred_lr, average='macro', zero_division=0)\n"
        'print(f"LR  Accuracy: {acc_lr:.4f}   Macro-F1: {f1_lr:.4f}")'
    ),
    md("## Challenger: Random Forest (100 trees)"),
    code(
        "from sklearn.ensemble import RandomForestClassifier\n"
        "rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,\n"
        "                            class_weight='balanced_subsample')\n"
        "rf.fit(X_tr_s, y_tr)\n"
        "y_pred_rf = rf.predict(X_te_s)\n"
        "acc_rf = accuracy_score(y_te, y_pred_rf)\n"
        "f1_rf  = f1_score(y_te, y_pred_rf, average='macro', zero_division=0)\n"
        'print(f"RF  Accuracy: {acc_rf:.4f}   Macro-F1: {f1_rf:.4f}")'
    ),
    md("## Comparison chart"),
    code(
        "import matplotlib.pyplot as plt\n"
        "models = ['Logistic Regression', 'Random Forest']\n"
        "accs   = [acc_lr, acc_rf]\n"
        "f1s    = [f1_lr, f1_rf]\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))\n"
        "ax1.bar(models, accs, color=['#5a7a9a', '#00d4ff'])\n"
        "ax1.set_title('Accuracy'); ax1.set_ylim(0, 1)\n"
        "ax2.bar(models, f1s, color=['#5a7a9a', '#a855f7'])\n"
        "ax2.set_title('Macro-F1'); ax2.set_ylim(0, 1)\n"
        "plt.suptitle('Baseline vs Random Forest'); plt.tight_layout(); plt.show()"
    ),
])

# ── 02 EDA ───────────────────────────────────────────────────────────────────

n02 = nb([
    md("# 02 · Exploratory Data Analysis — Security Log Analysis\n\n"
       "Full EDA: class distribution, feature statistics, correlations, "
       "and class-imbalance analysis on CICIDS2017."),
    code(
        "import sys; sys.path.insert(0, '..')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "from src.data.loader import load_raw, get_label_distribution, get_numeric_columns\n"
        "from src.data.preprocessing import clean_dataframe\n"
        "sns.set_theme(style='darkgrid', palette='muted')\n"
        'print("Ready")'
    ),
    md("## 1. Dataset overview (first 200k rows)"),
    code(
        "df = load_raw(nrows=200_000)\n"
        'print(f"Shape: {df.shape}")\n'
        'print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")\n'
        "df.head(3)"
    ),
    md("## 2. Full label distribution (entire dataset)"),
    code(
        "dist = get_label_distribution()\n"
        "labels = list(dist.keys()); counts = list(dist.values())\n"
        "total = sum(counts)\n"
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n"
        "palette = sns.color_palette('husl', len(labels))\n"
        "ax1.barh(labels, counts, color=palette)\n"
        "ax1.set_xlabel('Flow count'); ax1.set_title('All classes (linear scale)')\n"
        "ax2.barh(labels, [np.log10(max(c, 1)) for c in counts], color=palette)\n"
        "ax2.set_xlabel('log10(count)'); ax2.set_title('Log scale')\n"
        "plt.suptitle('CICIDS2017 — Label Distribution'); plt.tight_layout(); plt.show()\n"
        "for l, c in zip(labels, counts):\n"
        '    print(f"  {l:40s} {c:>10,}  ({c/total*100:5.2f}%)")'
    ),
    md("## 3. Feature statistics"),
    code(
        "num_cols = get_numeric_columns(df)\n"
        'print(f"Numeric features: {len(num_cols)}")\n'
        "df_clean = clean_dataframe(df, num_cols)\n"
        "df_clean[num_cols].describe().T.sort_values('std', ascending=False).head(20)"
    ),
    md("## 4. Correlation heatmap (top 20 by variance)"),
    code(
        "top20 = df_clean[num_cols].var().nlargest(20).index.tolist()\n"
        "corr = df_clean[top20].corr()\n"
        "fig, ax = plt.subplots(figsize=(14, 12))\n"
        "sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax, linewidths=0.3)\n"
        "ax.set_title('Correlation Matrix — Top 20 High-Variance Features')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 5. Flow duration by attack type"),
    code(
        "sample = df_clean.sample(30_000, random_state=42)\n"
        "fig, ax = plt.subplots(figsize=(14, 5))\n"
        "for label in sample['Label'].unique()[:8]:\n"
        "    vals = sample[sample['Label'] == label]['Flow Duration'].clip(0, 1e6)\n"
        "    ax.hist(vals, bins=60, alpha=0.5, label=label, density=True)\n"
        "ax.set_xlabel('Flow Duration (µs)'); ax.set_ylabel('Density')\n"
        "ax.set_title('Flow Duration Distribution by Attack Type (top 8)')\n"
        "ax.legend(fontsize=8); plt.tight_layout(); plt.show()"
    ),
    md("## 6. Class imbalance summary"),
    code(
        "benign = dist.get('BENIGN', 0)\n"
        "attack = total - benign\n"
        "ratio  = benign / total\n"
        'print(f"Benign flows : {benign:>10,}  ({ratio*100:.1f}%)")\n'
        'print(f"Attack flows : {attack:>10,}  ({(1-ratio)*100:.1f}%)")\n'
        'print(f"Imbalance    : {ratio / (1 - ratio):.1f} benign per attack flow")\n'
        "print(\"\\nMitigation: class_weight='balanced_subsample' in RandomForestClassifier.\")"
    ),
])

# ── 03 Random Forest Detection ────────────────────────────────────────────────

n03 = nb([
    md("# 03 · Random Forest Threat Detection\n\n"
       "Full RF pipeline: training, evaluation, feature importance, "
       "and confusion matrix visualisation."),
    code(
        "import sys; sys.path.insert(0, '..')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "from src.models.random_forest import is_trained, load_bundle, get_feature_importances\n"
        "from src.evaluation.metrics import load_model_report\n"
        'print(f"Model trained: {is_trained()}")'
    ),
    md("## Load the trained model bundle"),
    code(
        "if is_trained():\n"
        "    model, scaler, encoder, feat_cols = load_bundle()\n"
        '    print(f"Classes ({len(encoder.classes_)}): {encoder.classes_.tolist()}")\n'
        '    print(f"Features: {len(feat_cols)}")\n'
        "else:\n"
        "    print('Run: python src/train_random_forest.py')"
    ),
    md("## Evaluation report"),
    code(
        "report = load_model_report()\n"
        "if report:\n"
        '    print(f"Overall Accuracy : {report.get(\'accuracy\', 0):.4f}")\n'
        "    cr = report.get('classification_report', {})\n"
        "    wa = cr.get('weighted avg', {})\n"
        '    print(f"Weighted Avg F1  : {wa.get(\'f1-score\', 0):.4f}")\n'
        "    rows = [(k, v) for k, v in cr.items()\n"
        "            if k not in ('accuracy', 'macro avg', 'weighted avg')]\n"
        '    print(f"\\n{\'Class\':40s}  {\'Precision\':>9}  {\'Recall\':>6}  {\'F1\':>6}")\n'
        '    print("-" * 70)\n'
        "    for cls, vals in rows:\n"
        "        p = vals.get('precision', 0); r = vals.get('recall', 0); f = vals.get('f1-score', 0)\n"
        '        print(f"{cls:40s}  {p:9.4f}  {r:6.4f}  {f:6.4f}")\n'
        "else:\n"
        "    print('No report found — train model first.')"
    ),
    md("## Feature importance (top 20)"),
    code(
        "feats = get_feature_importances(top_n=20)\n"
        "if feats:\n"
        "    names  = [f['feature'] for f in feats]\n"
        "    values = [f['importance'] * 100 for f in feats]\n"
        "    fig, ax = plt.subplots(figsize=(10, 8))\n"
        "    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(names)))\n"
        "    ax.barh(names[::-1], values[::-1], color=colors)\n"
        "    ax.set_xlabel('Importance (%)')\n"
        "    ax.set_title('Top 20 Random Forest Feature Importances')\n"
        "    plt.tight_layout(); plt.show()\n"
        "else:\n"
        "    print('No importances found — train model first.')"
    ),
    md("## Quick prediction on a sample flow"),
    code(
        "if is_trained():\n"
        "    from src.models.random_forest import predict_single\n"
        "    sample_flow = {\n"
        "        'Destination Port': 80, 'Flow Duration': 109,\n"
        "        'Total Fwd Packets': 1, 'Total Backward Packets': 1,\n"
        "        'Flow Bytes/s': 110091.7, 'Flow Packets/s': 18348.6,\n"
        "        'Init_Win_bytes_forward': 29, 'Init_Win_bytes_backward': 256,\n"
        "    }\n"
        "    result = predict_single(sample_flow)\n"
        '    print(f"Prediction : {result[\'prediction\']}")\n'
        '    print(f"Confidence : {result[\'confidence\']*100:.2f}%")'
    ),
])

# ── 04 K-Means Clustering ─────────────────────────────────────────────────────

n04 = nb([
    md("# 04 · K-Means Attack Clustering\n\n"
       "Cluster CICIDS2017 network flows using K-Means to reveal hidden "
       "attack group structure. Includes elbow method and PCA visualisation."),
    code(
        "import sys; sys.path.insert(0, '..')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "from src.data.loader import get_stratified_sample, get_numeric_columns\n"
        "from src.data.preprocessing import clean_dataframe\n"
        "from src.models.kmeans import train_kmeans, get_cluster_profiles, elbow_scores\n"
        "from sklearn.preprocessing import StandardScaler\n"
        'print("Imports OK")'
    ),
    md("## Load and prepare 30 000-row sample"),
    code(
        "df = get_stratified_sample(n=30_000)\n"
        "numeric_cols = get_numeric_columns(df)\n"
        "df_clean = clean_dataframe(df, numeric_cols)\n"
        "X = df_clean[numeric_cols].values\n"
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(X)\n"
        'print(f"X shape: {X_scaled.shape}")'
    ),
    md("## Elbow method — choose optimal k"),
    code(
        "scores = elbow_scores(X_scaled, k_range=range(2, 11))\n"
        "ks       = [s['k']       for s in scores]\n"
        "inertias = [s['inertia'] for s in scores]\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.plot(ks, inertias, 'o-', color='#00d4ff', lw=2, ms=8)\n"
        "ax.set_xlabel('Number of clusters k'); ax.set_ylabel('Inertia')\n"
        "ax.set_title('Elbow Method — Optimal k Selection')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## Train K-Means with k = 8"),
    code(
        "labels_col = df['Label'] if 'Label' in df.columns else None\n"
        "result = train_kmeans(X_scaled, feature_names=numeric_cols, labels=labels_col, k=8)\n"
        'print(f"Silhouette score: {result[\'silhouette_score\']:.4f}")\n'
        'print("\\nCluster profiles:")\n'
        "for c in result['clusters']:\n"
        "    print(f\"  Cluster {c['cluster_id']:2d}: {c['size']:6,} flows | \"\n"
        "          f\"Dominant: {c['dominant_attack']}\")"
    ),
    md("## PCA 2D projection (5000 point scatter)"),
    code(
        "from sklearn.decomposition import PCA\n"
        "from sklearn.cluster import KMeans\n"
        "km = KMeans(n_clusters=8, random_state=42, n_init=10)\n"
        "cluster_labels = km.fit_predict(X_scaled)\n"
        "pca = PCA(n_components=2, random_state=42)\n"
        "X_2d = pca.fit_transform(X_scaled[:5000])\n"
        "cl_2d = cluster_labels[:5000]\n"
        "fig, ax = plt.subplots(figsize=(10, 7))\n"
        "sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=cl_2d, cmap='tab10', alpha=0.4, s=6)\n"
        "plt.colorbar(sc, ax=ax, label='Cluster ID')\n"
        "ax.set_title('K-Means Clusters — PCA 2D Projection')\n"
        "ax.set_xlabel('PC1'); ax.set_ylabel('PC2')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## Cluster composition heatmap"),
    code(
        "df['cluster'] = cluster_labels\n"
        "pivot = df.groupby(['cluster', 'Label']).size().unstack(fill_value=0)\n"
        "import seaborn as sns\n"
        "fig, ax = plt.subplots(figsize=(14, 5))\n"
        "sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=0.3)\n"
        "ax.set_title('Attack Type Count per Cluster')\n"
        "ax.set_xlabel('Attack Type'); ax.set_ylabel('Cluster ID')\n"
        "plt.tight_layout(); plt.show()"
    ),
])

# ── 05 Apriori Pattern Mining ─────────────────────────────────────────────────

n05 = nb([
    md("# 05 · Apriori Attack Pattern Mining\n\n"
       "Discover frequent itemsets and association rules from CICIDS2017 "
       "network flow transactions using the Apriori algorithm (mlxtend)."),
    code(
        "import sys; sys.path.insert(0, '..')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "from src.data.loader import get_stratified_sample\n"
        "from src.models.apriori import mine_patterns, get_top_rules\n"
        'print("Imports OK")'
    ),
    md("## Load 20 000-row sample"),
    code(
        "df = get_stratified_sample(n=20_000)\n"
        'print(f"Shape: {df.shape}")\n'
        "print(df['Label'].value_counts().head(8))"
    ),
    md("## Mine association rules\n\n"
       "- `min_support=0.05` → itemset appears in ≥ 5% of flows  \n"
       "- `min_confidence=0.6` → rule is correct ≥ 60% of the time  \n"
       "- Rules sorted by **lift** (lift > 1 means positive correlation)"),
    code(
        "rules = mine_patterns(df, min_support=0.05, min_confidence=0.6, max_rules=50)\n"
        'print(f"Rules discovered: {len(rules)}")\n'
        "if rules:\n"
        '    print(f"\\n{\'Antecedent\':40s}  {\'Consequent\':25s}  {\'Supp\':>5}  {\'Conf\':>5}  {\'Lift\':>6}")\n'
        '    print("-" * 95)\n'
        "    for r in rules[:10]:\n"
        "        ant = ' + '.join(r['antecedent'])\n"
        "        con = ', '.join(r['consequent'])\n"
        "        print(f\"{ant:40s}  {con:25s}  \"\n"
        "              f\"{r['support']:5.3f}  {r['confidence']:5.3f}  {r['lift']:6.3f}\")"
    ),
    md("## Support vs Confidence scatter (bubble size = lift)"),
    code(
        "if rules:\n"
        "    supp = [r['support']    for r in rules]\n"
        "    conf = [r['confidence'] for r in rules]\n"
        "    lift = [r['lift']       for r in rules]\n"
        "    sizes = [max(20, l * 60) for l in lift]\n"
        "    fig, ax = plt.subplots(figsize=(10, 6))\n"
        "    sc = ax.scatter(supp, conf, s=sizes, c=lift, cmap='plasma',\n"
        "                    alpha=0.7, edgecolors='w', lw=0.5)\n"
        "    plt.colorbar(sc, ax=ax, label='Lift')\n"
        "    ax.set_xlabel('Support'); ax.set_ylabel('Confidence')\n"
        "    ax.set_title('Apriori Rules: Support vs Confidence (bubble = Lift)')\n"
        "    ax.axhline(0.8, color='#00d4ff', ls='--', lw=1, label='Conf = 0.8')\n"
        "    ax.legend(); plt.tight_layout(); plt.show()\n"
        "else:\n"
        "    print('No rules — try lowering min_support.')"
    ),
    md("## Top rules as a DataFrame"),
    code(
        "top = get_top_rules(n=20)\n"
        "if top:\n"
        "    df_rules = pd.DataFrame(top)\n"
        "    df_rules['antecedent'] = df_rules['antecedent'].apply(lambda x: ' + '.join(x))\n"
        "    df_rules['consequent'] = df_rules['consequent'].apply(lambda x: ', '.join(x))\n"
        "    display(df_rules[['antecedent','consequent','support','confidence','lift']].head(20))\n"
        "else:\n"
        "    print('No cached rules found.')"
    ),
    md("## Lift distribution"),
    code(
        "if rules:\n"
        "    lifts = [r['lift'] for r in rules]\n"
        "    fig, ax = plt.subplots(figsize=(8, 4))\n"
        "    ax.hist(lifts, bins=20, color='#a855f7', edgecolor='white', lw=0.5)\n"
        "    ax.set_xlabel('Lift'); ax.set_ylabel('Number of rules')\n"
        "    ax.set_title('Lift Distribution Across Association Rules')\n"
        "    ax.axvline(1.0, color='#ff4757', ls='--', label='Lift = 1 (no assoc.)')\n"
        "    ax.legend(); plt.tight_layout(); plt.show()\n"
        '    print(f"Rules with lift > 2.0: {sum(1 for l in lifts if l > 2)}")'
    ),
])

# ── Write all notebooks ───────────────────────────────────────────────────────

notebooks = {
    "01_baseline_modeling.ipynb":       n01,
    "02_eda_security_analysis.ipynb":   n02,
    "03_random_forest_detection.ipynb": n03,
    "04_kmeans_attack_clustering.ipynb": n04,
    "05_apriori_attack_patterns.ipynb": n05,
}

for name, nb_data in notebooks.items():
    path = ROOT / name
    path.write_text(json.dumps(nb_data, indent=1), encoding="utf-8")
    print(f"Written: {name}  ({path.stat().st_size // 1024} KB)")

print("\nAll 5 notebooks created successfully.")
