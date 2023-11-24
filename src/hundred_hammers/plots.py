import numpy as np
import seaborn as sns
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from .metric_alias import metric_alias


def _save_plot(filepath, display, dpi=200):
    if filepath is not None:
        plt.savefig(filepath, dpi=dpi)
        if not display:
            plt.close()
    else:
        if display:
            plt.show()


def plot_confusion_matrix(X, y, model, class_dict, title="", test_size=0.2, seed=0, filepath=None, display=True):
    """
    Plot confusion matrix for a given model.

    :param X: input observations
    :param y: target values
    :param model: model to evaluate
    :param class_dict: dictionary with class names (ex.: {0: "No", 1: "Yes"})
    :param title: title of the plot
    :param test_size: percentage of the dataset to use for testing
    :param seed: random seed
    :param filepath: path to save the plot
    :param display: whether to display the plot
    """

    # Setting up
    labels = [class_dict[f] for f in np.unique(y)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=seed)

    # Plotting
    sns.set_style("ticks")
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.patch.set_facecolor("lightgrey")

    for i, md in enumerate([model, DummyClassifier()]):
        md.fit(X_train, y_train)

        cm_train = confusion_matrix(y_train, md.predict(X_train))
        cm_test = confusion_matrix(y_test, md.predict(X_test))

        for j, (cm_name, cm, _X, _y) in enumerate([("Train", cm_train, X_train, y_train), ("Test", cm_test, X_test, y_test)]):
            ax = axs[i, j]

            percentages = cm.astype("float") / cm.sum()
            annots = [[f"{v}\n\n({(p*100):.2f})" for v, p in zip(vs, ps)] for vs, ps in zip(cm, percentages)]
            cmap = sns.color_palette(["light:#7A7", "light:#77B"][j], as_cmap=True)

            sns.heatmap(
                cm,
                annot=annots,
                fmt="",
                square=True,
                xticklabels=labels,
                ax=ax,
                yticklabels=labels,
                linewidths=1,
                cmap=cmap,
                cbar=False,
                linecolor="black",
            )

            md_name = md.__class__.__name__
            acc = md.score(_X, _y)
            ax_title = f"{md_name} // {cm_name} data\n(N: {len(_y)}, Acc: {acc:.3f})"
            ax.set_title(ax_title, fontweight="bold")

            ax.set_xlabel("Predicted", fontweight="bold")
            ax.set_ylabel("Actual", fontweight="bold")

    plt.suptitle(title, fontweight="bold", fontsize=14)
    plt.tight_layout()

    _save_plot(filepath, display)


def plot_regression_pred(X, y, models, y_label="", title="", test_size=0.2, metric=None, seed=0, filepath=None, display=True):
    """
    Plot the predictions of the regression model

    :param X: input observations
    :param y: target values
    :param models: list of models to evaluate
    :param y_label: name of the target variable
    :param title: title of the plot
    :param test_size: percentage of the dataset to use for testing
    :param metric: metric to use for evaluation
    :param seed: random seed
    :param filepath: path to save the plot
    :param display: whether to display the plot
    """

    # Preparing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    _s = np.argsort(y_train)
    X_train, y_train = X_train[_s], y_train[_s]
    _s = np.argsort(y_test)
    X_test, y_test = X_test[_s], y_test[_s]

    _X = np.concatenate((X_train, X_test))
    _y = np.concatenate((y_train, y_test))

    # Plotting
    sns.set_style("ticks")

    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor("lightgrey")

    plt.plot(range(len(_y)), _y, "b", label="Data", linewidth=3)
    for md in models:
        md = md.fit(X_train, y_train)
        md_name = md.__class__.__name__

        res = metric(y_test, md.predict(X_test)) if metric is not None else ""
        metric_name = metric.__name__ if metric is not None else ""
        label = f"{md_name}" + (f"\n({metric_name}: {res:.2f})" if metric is not None else "")

        plt.plot(range(len(_y)), md.predict(_X), label=label)

    plt.axvline(len(y_train) - 0.5, color="k", linestyle="--", linewidth=3)
    plt.text(len(y_train) / 2, _y.max() * 0.9, "Train data", ha="center", va="center", fontsize=20)
    plt.text(len(y_train) + len(y_test) / 2, _y.max() * 0.9, "Test data", ha="center", va="center", fontsize=20)
    plt.gca().add_patch(Rectangle((len(y_train) - 0.5, _y.min()), len(y_test), (_y.max() - _y.min()), fill=True, alpha=0.1, color="b"))

    plt.title(title, fontweight="bold")
    plt.xlabel("Samples")
    plt.ylabel(y_label)
    plt.ylim(min(_y), max(_y))
    plt.xlim(0, len(_y))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    _save_plot(filepath, display)


def plot_batch_results(df, metric_name, title="", filepath=None, display=True):
    """
    Plot the results of the batch evaluation

    :param df: results dataframe
    :param title: title of plot
    :param filepath: filepath to save plot
    :param display: whether to display the plot
    """

    # sns.set_context("paper")
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    y_label = f"{metric_name} / Validation Test / Mean"
    x_label = f"{metric_name} / Validation Train / Mean"
    baseline_result = df[df["Model"].str.startswith("Dummy")][y_label].values[0]

    sns.scatterplot(data=df, x=x_label, y=y_label, s=100, hue="Model", legend=False)
    ax.axhline(baseline_result, color="grey", linestyle="--", linewidth=2, zorder=-1)
    ax.set_facecolor("#eeeeee")

    texts = []
    for name, row in df.iterrows():
        text = plt.text(row[x_label] + 0.005, row[y_label] + 0.001, row["Model"], fontsize=12)
        texts.append(text)
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5), force_text=1.0, force_points=1.0)

    plt.title(title, fontweight="bold")
    plt.grid()

    plt.tight_layout()

    _save_plot(filepath, display)


def plot_multiple_datasets(df, metric_name, id_col="Code", title="", line_at_0=False, higher_is_better=True, filepath=None, display=True):
    """
    Plot the results of the batch evaluation

    :param df: results dataframe
    :param metric_name: metric to plot
    :param id_col: column containing the ID of the dataset
    :param title: title of plot
    :param line_at_0: determines if a line is plotted at 0
    :param higher_is_better: determines if higher values are better
    :param filepath: filepath to save plot
    :param display: whether to display the plot
    """

    _df = df.sort_values(by=metric_name, ascending=(not higher_is_better))

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    sns.scatterplot(data=_df, x=id_col, y=metric_name, s=100, hue=id_col, legend=False)

    ax.set_xlabel("")
    ax.set_ylabel(metric_name)

    if line_at_0:
        ax.axhline(0, lw=4, color="k", zorder=-1)

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    ax.grid()
    ax.set_facecolor("#eeeeee")

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()

    _save_plot(filepath, display)
