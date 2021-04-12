import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, data, colors, interval=1):
        self.data = data
        self.colors = colors
        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots()
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=interval, 
                                          init_func=self.setup_plot, blit=True)


    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y = self.data[0, 0], self.data[1, 0]
        self.scat = self.ax.scatter(x, y, c=self.colors)

        self.ax.axis([-10, 10, -10, 10])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def update(self, i):
        """Update the scatter plot."""
        i = i % self.data.shape[1]
        x, y = self.data[0, i][..., np.newaxis], self.data[1, i][..., np.newaxis]
        self.scat.set_offsets(np.concatenate((x, y), axis=-1))

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,


