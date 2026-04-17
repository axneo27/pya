import matplotlib.pyplot as plt
import numpy as np


def plot_function():
    x = np.linspace(-5, 5, 500)
    y = x**2 * np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, color="blue", linewidth=2)
    ax.set_title("Графік функції sin(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    plt.show()


def histogram_normal():
    data = np.random.normal(loc=5, scale=2, size=1000)

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, color="skyblue", edgecolor="black")
    ax.set_title("Гістограма нормального розподілу (μ=5, σ=2)")
    ax.set_xlabel("Значення")
    ax.set_ylabel("Частота")
    plt.show()


def pie_hobbies():
    labels = ["rs", "читання", "відеоігри", "speedcubing", "спорт"]
    sizes = [40, 20, 15, 20, 5]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, shadow=True)
    ax.set_title("Мої хобі")
    ax.axis("equal")
    plt.show()


def boxplot_fruits():
    fruits = {
        "Яблука": np.random.normal(160, 15, 100),
        "Банани": np.random.normal(120, 10, 100),
        "Груші": np.random.normal(170, 12, 100),
        "Виноград": np.random.normal(5, 1.5, 100),
    }

    fig, ax = plt.subplots()
    ax.boxplot(list(fruits.values()), labels=list(fruits.keys())) # type: ignore
    ax.set_title("Box-plot маси фруктів")
    ax.set_ylabel("Маса (г)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.show()


def scatter_uniform():
    x = np.random.rand(100)
    y = np.random.rand(100)

    fig, ax = plt.subplots()
    ax.scatter(x, y, color="green", alpha=0.6)
    ax.set_title("Точкова діаграма")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", alpha=0.4)
    plt.show()


def multiple_functions():
    x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = 0.1 * x ** 2

    fig, ax = plt.subplots()
    ax.plot(x, y1, label="sin(x)", color="blue")
    ax.plot(x, y2, label="cos(x)", color="red")
    ax.plot(x, y3, label="0.1 x^2", color="green")
    ax.set_title("Функції")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()
    plt.show()


np.random.seed(42)
plot_function()
histogram_normal()
pie_hobbies()
boxplot_fruits()
scatter_uniform()
multiple_functions()
