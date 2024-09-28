import matplotlib.pyplot as plt


def plot(x: list, y: list, title: str, clear: bool = True) -> None:
    plt.plot(x, y)
    plt.savefig("plots/" + title + ".svg")

    if clear:
        plt.clf()
