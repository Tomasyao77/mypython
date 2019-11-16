import matplotlib.pyplot as plt


def plot(x, y, dict={"type": "line", "xlabel": "x", "ylabel": "y"}):
    if dict is None:
        return 0
    check = ["type", "xlabel", "ylabel"]
    dict_keys = dict.keys()
    for i in range(check.__len__()):
        if check[i] not in dict_keys:
            return 1

    if dict["type"] == "line":
        # 折线图 line
        plt.xlabel(dict["xlabel"])
        plt.ylabel(dict["ylabel"])
        plt.plot(x, y)
    elif dict["type"] == "bar":
        # 直方图 bar
        plt.xlabel(dict["xlabel"])
        plt.ylabel(dict["ylabel"])
        plt.bar(x, y)
    plt.show()
