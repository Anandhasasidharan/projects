import seaborn as sns
import matplotlib.pyplot as plt

def plot_proto_vs_severity(df):
    plt.figure(figsize=(10, 6))
    proto_mean = df.groupby('proto')['severity_score'].mean().sort_values()
    sns.barplot(x=proto_mean.values, y=proto_mean.index, palette="viridis")
    plt.xlabel("Avg Severity Score")
    plt.ylabel("Protocol")
    plt.title("Avg Severity by Protocol")
    plt.tight_layout()
    plt.show()

def plot_box_by_category(df, col='proto'):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=col, y='severity_score', order=df[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f"Severity Score by {col}")
    plt.tight_layout()
    plt.show()

def plot_stripplot_by_category(df, col='proto', sample_size=1000):
    plt.figure(figsize=(12, 6))
    sns.stripplot(data=df.sample(sample_size), x=col, y='severity_score', jitter=True, size=3, alpha=0.3)
    plt.xticks(rotation=45)
    plt.title(f"Severity Score Stripplot by {col}")
    plt.tight_layout()
    plt.show()
