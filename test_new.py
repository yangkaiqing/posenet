import numpy as np
import tensorflow as tf
from posenet import GoogLeNet as PoseNet
import cv2
from tqdm import tqdm
import os

batch_size = 1
mean_train = np.loadtxt('./data1/mean.csv')
std_train = np.loadtxt('./data1/std.csv')
min_train = np.loadtxt('./data1/min.csv')
max_train = np.loadtxt('./data1/max.csv')

path = './M/sence7_loss/PoseNet_best.ckpt'
test_dir = '/home/dusen/YKQ/tensorflow-posenet-master/posenet/scene7_jiading_riverside_test'
result_dir = './result/' + 'pose_estim1.csv'
test_image_list = os.listdir(test_dir)


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
    images_out = []  # final result
    # Resize and crop and compute mean!
    images_cropped = []
    for i in tqdm(range(len(images))):
        every_image_concat = []
        img = cv2.imread(images[i])
        image_resize = cv2.resize(img, (480, 240))
        image_center_crop = centeredCrop(image_resize, 224)
        every_image_concat.append(np.expand_dims(image_center_crop, axis=0))
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
                every_image_concat.append(np.expand_dims(cropped_img, axis=0))
                num = num + 1
        np_every_image_concat = np.concatenate(every_image_concat, axis=3)
        images_cropped.append(np_every_image_concat)
    images_out = images_cropped
    return images_out
def main():
    image = tf.placeholder(tf.float32, [1, 224, 224, 30])
    net = PoseNet({'data': image})
    p3_x = net.layers['cls3_fc_pose_xyz']
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, path)
        scene_dict = {}
        id_list = list()
        images = []
        K = []
        for each in test_image_list:
            images.append(test_dir + "/" + each + '/thumbnail.jpg')
            K.append(each)
        images_val = preprocess(images)
        index = 0
        for img in images_val:
            np_image = img
            predicted_x = sess.run([p3_x], feed_dict={image: np_image})
            predicted_x = np.squeeze(predicted_x) / 100.0 * (max_train - min_train) + min_train
            scene_dict[K[index]] = predicted_x
            index = index + 1
        for id in scene_dict.keys():
            id_list.append(id)
        id_list1 = sorted(id_list)
    with open(result_dir, 'w') as csv_file:
        for k in id_list1:
            line = k + ',' + str('%.4f' % scene_dict[k][0]) + ',' + str('%.4f' % scene_dict[k][1]) + ',' + str(
                '%.4f' % scene_dict[k][2]) + '\n'
            print(line)
            csv_file.write(line)


if __name__ == '__main__':
    main()
