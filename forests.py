# python data science pg 421-432
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; 
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from visualize_classifier import visualize_classifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, RandomForestRegressor

sns.set()


X, y = make_blobs(n_samples=300, centers=4,
random_state=0, cluster_std=1.0)
#plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
#plt.show()

# fit the decision tree to the datas

tree = DecisionTreeClassifier().fit(X, y)
visualize_classifier(DecisionTreeClassifier(), X, y)

#random forests
tree = DecisionTreeClassifier()
bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8,
random_state=1)
bag.fit(X, y)
visualize_classifier(bag, X, y)

model = RandomForestClassifier(n_estimators=100, random_state=0)
visualize_classifier(model, X, y)

#random forest regression
rng = np.random.RandomState(42)
x = 10 * rng.rand(200)

def model(x, sigma=0.3):
    fast_oscillation = np.sin(5 * x)
    slow_oscillation = np.sin(0.5 * x)
    noise = sigma * rng.randn(len(x))
    return slow_oscillation + fast_oscillation + noise

y = model(x)
plt.errorbar(x, y, 0.3, fmt='o')

forest = RandomForestRegressor(200)
forest.fit(x[:, None], y)
xfit = np.linspace(0, 10, 1000)
yfit = forest.predict(xfit[:, None])
ytrue = model(xfit, sigma=0)
plt.errorbar(x, y, 0.3, fmt='o', alpha=0.5)
plt.plot(xfit, yfit, '-r')
plt.plot(xfit, ytrue, '-k', alpha=0.5)
plt.show()