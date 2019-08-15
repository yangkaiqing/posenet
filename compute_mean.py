import numpy as np
import os
directory_train = './data1/scene7_jiading_riverside_training'
nums = len(os.listdir(directory_train))
print(nums-1)
dataset = '/jiading_riverside_training_coordinates.csv'
f_train = open(directory_train + dataset)
next(f_train)
pose_train = np.zeros((nums, 3))
j = 0
for train_line in f_train:
    fname_train, p0, p1, p2 = train_line.split(',')
    p0 = float(p0)
    p1 = float(p1)
    p2 = float(p2)
    pose_train[j, :] = np.squeeze([p0, p1, p2])
    j = j + 1
mean = np.mean(pose_train, axis=0)
max = np.max(pose_train, axis=0)
min = np.min(pose_train, axis=0)
std = np.std(pose_train, axis=0)
np.savetxt('./data1/mean.csv', mean, fmt='%4f')
np.savetxt('./data1/max.csv', max, fmt='%4f')
np.savetxt('./data1/min.csv', min, fmt='%4f')
np.savetxt('./data1/std.csv', std, fmt='%4f')
