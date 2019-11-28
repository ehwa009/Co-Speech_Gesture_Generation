import matplotlib.pyplot as plt
import numpy as np

from matplotlib import pyplot, transforms
from matplotlib import animation, rc

from matplotlib.animation import FFMpegWriter
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

class Plot():

    def __init__(self, x_lim, y_lim):
        """Instantiates an object to visualize the generated poses."""
        self.fig = plt.figure()
        ax = plt.axes(xlim=x_lim, ylim=y_lim)

        base = pyplot.gca().transData
        rot = transforms.Affine2D().rotate_deg(180)
        
        [self.line] = ax.plot([], [], lw=5, transform=rot+base)
        
        [self.line2] = ax.plot([], [], lw=5, transform=rot+base)
        [self.line3] = ax.plot([], [], lw=5, transform=rot+base)
        [self.line4] = ax.plot([], [], lw=5, transform=rot+base)

        [self.line5] = ax.plot([], [], lw=5, transform=rot+base)
        [self.line6] = ax.plot([], [], lw=5, transform=rot+base)
        [self.line7] = ax.plot([], [], lw=5, transform=rot+base)

    def init_line(self):
        """Creates line objects which are drawn later."""
        self.line.set_data([], [])
        
        self.line2.set_data([], [])
        self.line3.set_data([], [])
        self.line4.set_data([], [])

        self.line5.set_data([], [])
        self.line6.set_data([], [])
        self.line7.set_data([], [])
        
        return ([self.line, self.line2, self.line3, self.line4, self.line5, self.line6, self.line7])

    def animate_frame(self, pose):
        self.line.set_data(pose[0:6:3], pose[1:6:3])
        
        self.line2.set_data(pose[3:9:3], pose[4:9:3])
        self.line3.set_data([pose[6], pose[9]], [pose[7], pose[10]])
        self.line4.set_data([pose[12], pose[9]], [pose[13], pose[10]])
        
        self.line5.set_data([pose[3],pose[15]], [pose[4],pose[16]])
        self.line6.set_data([pose[15], pose[18]], [pose[16], pose[19]])
        self.line7.set_data([pose[21], pose[18]], [pose[22], pose[19]])

        return ([self.line, self.line2, self.line3, self.line4, self.line5, self.line6, self.line7])

    def animate(self, frames_to_play, interval):
        """Returns a matplotlib animation object that can be saved as a video."""
        anim = animation.FuncAnimation(self.fig, self.animate_frame, 
                                        init_func=self.init_line, frames=frames_to_play, 
                                        interval=interval, blit=True)

        return anim

    
    def display_pose(self, pose, degree=0, linewidth=5.0):
        base = pyplot.gca().transData
        rot = transforms.Affine2D().rotate_deg(degree)

        plt.plot(pose[0:6:3], pose[1:6:3], transform=rot+base, linewidth=linewidth) #neck
        plt.plot(pose[3:9:3], pose[4:9:3], transform=rot+base,linewidth=linewidth) #sholder 1
        plt.plot([pose[6], pose[9]], [pose[7], pose[10]], transform=rot+base, linewidth=linewidth)
        plt.plot([pose[12], pose[9]], [pose[13], pose[10]], transform=rot+base, linewidth=linewidth) #arm1-1

        plt.plot([pose[3],pose[15]], [pose[4],pose[16]], transform=rot+base, linewidth=linewidth) #sholder 2
        plt.plot([pose[15], pose[18]], [pose[16], pose[19]], transform=rot+base, linewidth=linewidth)
        plt.plot([pose[21], pose[18]], [pose[22], pose[19]], transform=rot+base, linewidth=linewidth) #arm2-1

    
    def display_multi_poses(self, poses, col=10):
        fig = plt.figure(figsize=(30, 9))
        fig.subplots_adjust(hspace=0.4, wspace=0.5)
        
        row = poses.shape[0] / col
        
        for i in range(1, poses.shape[0]+1):
            plt.subplot(row, col, i)
            self.display_pose(poses[i-1], linewidth=3.0)

    def plot_confusion_matrix(self, y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    def save(self, ani, name):
        writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('{}'.format(name), writer=writer)
        print("{}.mp4 file saved.".format(name))

        