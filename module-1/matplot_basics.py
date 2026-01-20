import numpy as np

import matplotlib.pyplot as plt


def basic_line_plot():
    # Simple line plot: y = x^2
    x = np.linspace(-10, 10, 100)
    y = x ** 2

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label="y = x^2", color="blue", linewidth=2)
    plt.title("Basic Line Plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def multiple_lines_plot():
    x = np.linspace(0, 2 * np.pi, 200)
    sin_y = np.sin(x)
    cos_y = np.cos(x)
    
    print(sin_y)
    print(cos_y)
    

    plt.figure(figsize=(6, 4))
    plt.plot(x, sin_y, label="sin(x)", color="red")
    plt.plot(x, cos_y, label="cos(x)", color="green")
    plt.title("Multiple Lines Plot")
    plt.xlabel("x")
    plt.ylabel("value")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def scatter_plot():
    # Scatter plot with random data
    np.random.seed(0)
    x = np.random.rand(50)
    y = np.random.rand(50)
    colors = np.random.rand(50)
    sizes = 1000 * np.random.rand(50)

    plt.figure(figsize=(6, 4))
    scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.7, cmap="viridis")
    plt.title("Basic Scatter Plot")
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.colorbar(scatter, label="Color scale")
    plt.tight_layout()
    plt.show()


def bar_plot():
    categories = ["A", "B", "C", "D"]
    values = [10, 24, 36, 18]

    plt.figure(figsize=(6, 4))
    plt.bar(categories, values, color=["skyblue", "orange", "green", "red"])
    plt.title("Basic Bar Plot")
    plt.xlabel("Category")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()


def histogram_plot():
    np.random.seed(1)
    data = np.random.normal(loc=0, scale=1, size=1000)

    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=30, color="purple", alpha=0.7, edgecolor="black")
    plt.title("Histogram of Normal Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def subplots_example():
    x = np.linspace(0, 10, 100)
    y1 = x
    y2 = x ** 2
    y3 = np.sqrt(x + 1e-9)
    y4 = np.log(x + 1)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes[0, 0].plot(x, y1, color="blue")
    axes[0, 0].set_title("y = x")

    axes[0, 1].plot(x, y2, color="red")
    axes[0, 1].set_title("y = x^2")

    axes[1, 0].plot(x, y3, color="green")
    axes[1, 0].set_title("y = sqrt(x)")

    axes[1, 1].plot(x, y4, color="orange")
    axes[1, 1].set_title("y = log(x + 1)")

    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Subplots Example", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Call any function you want to demo
    basic_line_plot()
    multiple_lines_plot()
    scatter_plot()
    bar_plot()
    histogram_plot()
    subplots_example()