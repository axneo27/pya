import matplotlib.pyplot as plt
import numpy as np

# графіків функцій
x = np.linspace(-10, 10, 500)
y = np.sin(x)
# y = x ** 2

plt.plot(x, y)
plt.title("Графік функції sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()

# гістограм розподілу змінних
data = np.random.randn(1000)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Гістограма розподілу")
plt.xlabel("Значення")
plt.ylabel("Частота")
plt.show()

# Pie Chart
labels = ['Python', 'Java', 'C++', 'JavaScript']
sizes = [40, 25, 20, 15]

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Популярність мов програмування")
plt.axis('equal')  # рівні осі для круга
plt.show()

# Scatter plot
x = np.random.rand(100)
y = np.random.rand(100)

plt.scatter(x, y, color='green')
plt.title("Точкова діаграма")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Bar-plot діаграми
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 12, 36]

plt.bar(categories, values, color='orange')
plt.title("Bar Plot приклад")
plt.xlabel("Категорія")
plt.ylabel("Значення")
plt.show()

# xtiks rotate
##generating some data
years = [1936, 1945]+[i for i in range(1947,1997)]
data1 = np.random.rand(len(years))
data2 = np.random.rand(len(years))

diabete = {key: val for key,val in zip(years, data1)}
not_diabete = {key: val for key,val in zip(years, data2)}

##the actual graph:
fig, ax = plt.subplots(figsize = (10,4))

idx = np.asarray([i for i in range(len(years))])

width = 0.2

ax.bar(idx, [val for key,val in sorted(diabete.items())], width=width)
ax.bar(idx+width, [val for key,val in sorted(not_diabete.items())], width=width)

ax.set_xticks(idx)
ax.set_xticklabels(years, rotation=65) # type: ignore
ax.legend(['Diabete', 'Non-Diabete'])
ax.set_xlabel('years')
ax.set_ylabel('# of patients')

fig.tight_layout()

plt.show()