import matplotlib.pyplot as plt

# visualization
def lin_reg_plot(X, y, model):
    plt.scatter(X, y, c = 'steelblue', edgecolor = 'white', s = 70)
    plt.plot(X, model.predict(X), color = 'black', lw = 2)

    return