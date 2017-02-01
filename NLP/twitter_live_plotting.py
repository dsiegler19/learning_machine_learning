from matplotlib import pyplot
from matplotlib import animation
from matplotlib import style
import os

style.use("ggplot")

fig = pyplot.figure()
ax1 = fig.add_subplot(1, 1, 1)

def animate(i):
    pullData = open(os.getcwd() + "/twitter-output.txt", "r").read()
    lines = pullData.split('\n')

    xar = []
    yar = []

    x = 0
    y = 0

    for l in lines[-200:]:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 1

        xar.append(x)
        yar.append(y)

    ax1.clear()
    ax1.plot(xar, yar)

ani = animation.FuncAnimation(fig, animate, interval=1000)
pyplot.show()