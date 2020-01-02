import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
import pandas as pd

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
        
        [self.line] = ax.plot([], [], lw=5)
        
        [self.line2] = ax.plot([], [], lw=5)
        [self.line3] = ax.plot([], [], lw=5)
        [self.line4] = ax.plot([], [], lw=5)

        [self.line5] = ax.plot([], [], lw=5)
        [self.line6] = ax.plot([], [], lw=5)
        [self.line7] = ax.plot([], [], lw=5)

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

    def save(self, ani, name):
        writer = FFMpegWriter(fps=18, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('{}'.format(name), writer=writer)
        print("[INFO] {}.mp4 file saved.".format(name))


def display_pose(pose, degree=0, linewidth=5.0):
    base = pyplot.gca().transData
    rot = transforms.Affine2D().rotate_deg(degree)

    plt.plot(pose[0:6:3], pose[1:6:3], transform=rot+base, linewidth=linewidth) #neck
    plt.plot(pose[3:9:3], pose[4:9:3], transform=rot+base,linewidth=linewidth) #sholder 1
    plt.plot([pose[6], pose[9]], [pose[7], pose[10]], transform=rot+base, linewidth=linewidth)
    plt.plot([pose[12], pose[9]], [pose[13], pose[10]], transform=rot+base, linewidth=linewidth) #arm1-1

    plt.plot([pose[3],pose[15]], [pose[4],pose[16]], transform=rot+base, linewidth=linewidth) #sholder 2
    plt.plot([pose[15], pose[18]], [pose[16], pose[19]], transform=rot+base, linewidth=linewidth)
    plt.plot([pose[21], pose[18]], [pose[22], pose[19]], transform=rot+base, linewidth=linewidth) #arm2-1


def display_multi_poses(poses, col=10):
    row = poses.shape[0] / col
    fig = plt.figure(figsize=(row, col))
    fig.subplots_adjust(hspace=0.4, wspace=0.5)
    
    for i in range(1, poses.shape[0]+1):
        plt.subplot(row, col, i)
        display_pose(poses[i-1], linewidth=3.0)


def display_loss(log_train_file, log_vaild_file):
    loss_tr = []
    loss_vf = []
    with open(log_train_file, 'r') as log_tf, open(log_vaild_file, 'r') as log_vf:
        for l in log_tf:
            line = l.rstrip()
            line = line.split(',')
            val = line[1].strip()
            try:
                loss_tr.append(float(line[1].strip()))
            except ValueError:
                pass
        for l in log_vf:
            line = l.rstrip()
            line = line.split(',')
            val = line[1].strip()
            try:
                loss_vf.append(float(line[1].strip()))
            except ValueError:
                pass

    loss_df = pd.DataFrame({
        'Epoch': [i for i in range(len(loss_tr))],
        'Train_loss': loss_tr,
        'Valid_loss': loss_vf
    })  
    #print(loss_df.head())
    #exit(-1)
    plt.figure(figsize=(15, 9))
    sns.set_style('darkgrid')
    sns.lineplot(data=loss_df, x='Epoch', y='Train_loss', label='Train')
    sns.lineplot(data=loss_df, x='Epoch', y='Valid_loss', label='Valid')
    
#    plt.plot(loss_tr, '-r', label='train')
 #   plt.plot(loss_vf, '-b', label='validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train vs. Valid loss')

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('-model', default='seq2pos')
    parser.add_argument('-model', default='transformer')
    parser.add_argument('-log', default='./log/')
    opt = parser.parse_args()

    display_loss(opt.log+opt.model+'_train.log', opt.log+opt.model+'_valid.log')
    plt.savefig(opt.log+'loss.png')


if __name__=='__main__':
    main()