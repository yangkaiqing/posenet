import numpy as np
import random
import tensorflow as tf
from posenet import GoogLeNet as PoseNet
import cv2
from tqdm import tqdm
import time

time_start=time.time()

batch_size = 10
max_iterations = 1100
# Set this path to your dataset directory
directory = './data1/scene7_jiading_riverside_training'
dataset = '/jiading_riverside_training_coordinates.csv'
outputFile = "./M/sence7_loss/"

mean_train = np.loadtxt('./data1/mean.csv')
min_train = np.loadtxt('./data1/min.csv')
max_train = np.loadtxt('./data1/max.csv')


class datasource(object):
    def __init__(self, images, poses):
        self.images = images
        self.poses = poses


def centeredCrop(img, output_side_length):
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    height_offset = (new_height - output_side_length) / 2
    width_offset = (new_width - output_side_length) / 2
    cropped_img = img[height_offset:height_offset + output_side_length,
                  width_offset:width_offset + output_side_length]
    return cropped_img


def preprocess(images):
    images_cropped = []
    for i in tqdm(range(len(images))):
        every_image_concat = []
        img = cv2.imread(images[i])
        image_resize = cv2.resize(img, (480, 240))
        image_center_crop = centeredCrop(image_resize, 224)
        every_image_concat.append(image_center_crop)
        w = 480
        h = 240
        d = 32
        hang = (w - d) // d
        lie = (h - d) // d
        num = 0
        img = cv2.imread(images[i])
        image_resize_2 = cv2.resize(img, (480, 240))
        for i in range(lie):
            for j in range(hang):
                if num > 8:
                    break
                cropped_img = image_resize_2[i * d:i * d + 224, j * d:j * d + 224, :]
                every_image_concat.append(cropped_img)
                num = num + 1
        np_every_image_concat = np.concatenate(every_image_concat,axis=2)
        images_cropped.append(np_every_image_concat)
    images_out = images_cropped
    return images_out


def get_data():
    poses = []
    images = []
    with open(directory + dataset) as f:
        next(f)
        for line in f:
            fname, p0, p1, p2 = line.split(',')
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            poses.append((p0, p1, p2))
            images.append(directory + "/" + fname + '/' + 'thumbnail.jpg')

    images = preprocess(images)
    return datasource(images, poses)


def gen_data(source):
    while True:
        indices = range(len(source.images))
        random.shuffle(indices)
        for i in indices:
            image = source.images[i]
            pose_x = source.poses[i]
            pose_x = ((pose_x - min_train)/(max_train - min_train))*100.0
            yield image, pose_x


def gen_data_batch(source):
    data_gen = gen_data(source)
    while True:
        image_batch = []
        pose_x_batch = []
        for _ in range(batch_size):
            image, pose_x, = next(data_gen)
            image_batch.append(image)
            pose_x_batch.append(pose_x)
        yield np.array(image_batch), np.array(pose_x_batch)


def main():
    images = tf.placeholder(tf.float32, [batch_size, 224, 224, 30])
    poses_x = tf.placeholder(tf.float32, [batch_size, 3])

    datasource = get_data()
    net = PoseNet({'data': images})
    p1_x = net.layers['cls1_fc_pose_xyz']

    p2_x = net.layers['cls2_fc_pose_xyz']

    p3_x = net.layers['cls3_fc_pose_xyz']

    l1_x = tf.reduce_mean(tf.square(tf.subtract(p1_x, poses_x))) * 0.3
    l2_x = tf.reduce_mean(tf.square(tf.subtract(p2_x, poses_x))) * 0.3
    l3_x = tf.reduce_mean(tf.square(tf.subtract(p3_x, poses_x))) * 1
    loss = l1_x + l2_x + l3_x

    global_ = tf.Variable(tf.constant(0))
    lr = tf.train.exponential_decay(0.0001, global_, 500, 0.1, staircase=False)
    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False,
                                 name='Adam').minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        min_loss = 100

        data_gen = gen_data_batch(datasource)
        for i in range(max_iterations):
            sess.run(lr, feed_dict={global_: i})
            np_images, np_poses_x = next(data_gen)
            feed = {images: np_images, poses_x: np_poses_x}

            sess.run(opt, feed_dict=feed)
            np_loss = sess.run(loss, feed_dict=feed)
            if i % 20 == 0:
                print("iteration: " + str(i) + "\n\t" + "Loss is: " + str(np_loss))

            if np_loss < min_loss:
                min_loss = np_loss
                saver.save(sess, outputFile + 'PoseNet_best.ckpt')
                print("loss minest model saved: " + outputFile + 'PoseNet_best.ckpt')

            if i % 1000 == 0 or i == max_iterations:
                saver.save(sess, outputFile + 'PoseNet'+'_'+str(i) +'.ckpt')
                print("Intermediate file saved at: " + outputFile + 'PoseNet'+'_'+str(i) +'.ckpt')
        saver.save(sess, outputFile + 'PoseNet' + '_' + str(i) + '.ckpt')
        print("Intermediate file saved at: " + outputFile + 'PoseNet' + '_' + str(i) + '.ckpt')
    time_end=time.time()
    print('totally cost', time_end-time_start)

if __name__ == '__main__':
    main()
