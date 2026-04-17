import matplotlib.pyplot as plt
import numpy as np # type: ignore

# Generate sample data
group1 = np.random.normal(70, 10, 100)
group2 = np.random.normal(75, 7, 100)
group3 = np.random.normal(65, 12, 100)

# Combine data into list
data = [group1, group2, group3]

# Box plot with labels
plt.boxplot(data, labels=["Group A", "Group B", "Group C"]) # type: ignore
plt.title("Scores by Group")
plt.xlabel("Group")
plt.ylabel("Score")
plt.grid(True)
plt.show()
