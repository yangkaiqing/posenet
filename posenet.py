from network import Network

class GoogLeNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 640, 2, 2, name='conv1', group=10)
             .max_pool(3, 3, 2, 2, name='pool1')
             .lrn(2, 2e-05, 0.75, name='norm1')
             .conv(1, 1, 640, 1, 1, name='reduction2', group=10)
             .conv(3, 3, 1920, 1, 1, name='conv2', group=10)
             .lrn(2, 2e-05, 0.75, name='norm2')
             .max_pool(3, 3, 2, 2, name='pool2')
             .conv(1, 1, 960, 1, 1, name='icp1_reduction1', group=10)
             .conv(3, 3, 1280, 1, 1, name='icp1_out1', group=10))

        (self.feed('pool2')
             .conv(1, 1, 160, 1, 1, name='icp1_reduction2', group=10)
             .conv(5, 5, 320, 1, 1, name='icp1_out2', group=10))

        (self.feed('pool2')
             .max_pool(3, 3, 1, 1, name='icp1_pool')
             .conv(1, 1, 320, 1, 1, name='icp1_out3', group=10))

        (self.feed('pool2')
             .conv(1, 1, 640, 1, 1, name='icp1_out0', group=10))

        (self.feed('icp1_out0',
                   'icp1_out1',
                   'icp1_out2',
                   'icp1_out3')
             .concat(3, name='icp2_in')
             .conv(1, 1, 1280, 1, 1, name='icp2_reduction1', group=10)
             .conv(3, 3, 1920, 1, 1, name='icp2_out1', group=10))

        (self.feed('icp2_in')
             .conv(1, 1, 320, 1, 1, name='icp2_reduction2', group=10)
             .conv(5, 5, 960, 1, 1, name='icp2_out2', group=10))

        (self.feed('icp2_in')
             .max_pool(3, 3, 1, 1, name='icp2_pool')
             .conv(1, 1, 640, 1, 1, name='icp2_out3', group=10))

        (self.feed('icp2_in')
             .conv(1, 1, 1280, 1, 1, name='icp2_out0', group=10))

        (self.feed('icp2_out0',
                   'icp2_out1',
                   'icp2_out2',
                   'icp2_out3')
             .concat(3, name='icp2_out')
             .max_pool(3, 3, 2, 2, name='icp3_in')
             .conv(1, 1, 960, 1, 1, name='icp3_reduction1', group=10)
             .conv(3, 3, 2080, 1, 1, name='icp3_out1', group=10))

        (self.feed('icp3_in')
             .conv(1, 1, 160, 1, 1, name='icp3_reduction2', group=10)
             .conv(5, 5, 480, 1, 1, name='icp3_out2', group=10))

        (self.feed('icp3_in')
             .max_pool(3, 3, 1, 1, name='icp3_pool')
             .conv(1, 1, 640, 1, 1, name='icp3_out3', group=10))

        (self.feed('icp3_in')
             .conv(1, 1, 1920, 1, 1, name='icp3_out0', group=10))

        (self.feed('icp3_out0',
                   'icp3_out1',
                   'icp3_out2',
                   'icp3_out3')
             .concat(3, name='icp3_out')
             .avg_pool(5, 5, 3, 3, padding='VALID', name='cls1_pool')
             .conv(1, 1, 128, 1, 1, name='cls1_reduction_pose')
             #.dwise_conv(k_h=5, k_w=5, channel_multiplier=1, strides=[1, 1, 1, 1], padding='SAME', stddev=0.002, name='dwise_conv1')
             #.conv(5, 5, 128, 1, 1, relu=True, name='dw1')
             #.conv(1, 1, 128, 1, 1, relu=False,name='pw1')
             .fc(1024, name='cls1_fc1_pose')
             .fc(3, relu=False, name='cls1_fc_pose_xyz'))



        (self.feed('icp3_out')
             .conv(1, 1, 112, 1, 1, name='icp4_reduction1')
             .conv(3, 3, 224, 1, 1, name='icp4_out1'))

        (self.feed('icp3_out')
             .conv(1, 1, 24, 1, 1, name='icp4_reduction2')
             .conv(5, 5, 64, 1, 1, name='icp4_out2'))

        (self.feed('icp3_out')
             .max_pool(3, 3, 1, 1, name='icp4_pool')
             .conv(1, 1, 64, 1, 1, name='icp4_out3'))

        (self.feed('icp3_out')
             .conv(1, 1, 160, 1, 1, name='icp4_out0'))

        (self.feed('icp4_out0',
                   'icp4_out1',
                   'icp4_out2',
                   'icp4_out3')
             .concat(3, name='icp4_out')
             .conv(1, 1, 128, 1, 1, name='icp5_reduction1')
             .conv(3, 3, 256, 1, 1, name='icp5_out1'))

        (self.feed('icp4_out')
             .conv(1, 1, 24, 1, 1, name='icp5_reduction2')
             .conv(5, 5, 64, 1, 1, name='icp5_out2'))

        (self.feed('icp4_out')
             .max_pool(3, 3, 1, 1, name='icp5_pool')
             .conv(1, 1, 64, 1, 1, name='icp5_out3'))

        (self.feed('icp4_out')
             .conv(1, 1, 128, 1, 1, name='icp5_out0'))

        (self.feed('icp5_out0',
                   'icp5_out1',
                   'icp5_out2',
                   'icp5_out3')
             .concat(3, name='icp5_out')
             .conv(1, 1, 144, 1, 1, name='icp6_reduction1')
             .conv(3, 3, 288, 1, 1, name='icp6_out1'))

        (self.feed('icp5_out')
             .conv(1, 1, 32, 1, 1, name='icp6_reduction2')
             .conv(5, 5, 64, 1, 1, name='icp6_out2'))

        (self.feed('icp5_out')
             .max_pool(3, 3, 1, 1, name='icp6_pool')
             .conv(1, 1, 64, 1, 1, name='icp6_out3'))

        (self.feed('icp5_out')
             .conv(1, 1, 112, 1, 1, name='icp6_out0'))

        (self.feed('icp6_out0',
                   'icp6_out1',
                   'icp6_out2',
                   'icp6_out3')
             .concat(3, name='icp6_out')
             .avg_pool(5, 5, 3, 3, padding='VALID', name='cls2_pool')
             .conv(1, 1, 128, 1, 1, name='cls2_reduction_pose')
             #.conv(5, 5, 128, 1, 1, relu=True, name='dw2')
             #.dwise_conv(k_h=5, k_w=5, channel_multiplier=1, strides=[1, 1, 1, 1], padding='SAME', stddev=0.002,name='dwise_conv2')
             #.conv(1, 1, 128, 1, 1, relu=False,name='pw2')
             .fc(1024, name='cls2_fc1')
             .fc(3, relu=False, name='cls2_fc_pose_xyz'))



        (self.feed('icp6_out')
             .conv(1, 1, 160, 1, 1, name='icp7_reduction1')
             .conv(3, 3, 320, 1, 1, name='icp7_out1'))

        (self.feed('icp6_out')
             .conv(1, 1, 32, 1, 1, name='icp7_reduction2')
             .conv(5, 5, 128, 1, 1, name='icp7_out2'))

        (self.feed('icp6_out')
             .max_pool(3, 3, 1, 1, name='icp7_pool')
             .conv(1, 1, 128, 1, 1, name='icp7_out3'))

        (self.feed('icp6_out')
             .conv(1, 1, 256, 1, 1, name='icp7_out0'))

        (self.feed('icp7_out0',
                   'icp7_out1',
                   'icp7_out2',
                   'icp7_out3')
             .concat(3, name='icp7_out')
             .max_pool(3, 3, 2, 2, name='icp8_in')
             .conv(1, 1, 160, 1, 1, name='icp8_reduction1')
             .conv(3, 3, 320, 1, 1, name='icp8_out1'))

        (self.feed('icp8_in')
             .conv(1, 1, 32, 1, 1, name='icp8_reduction2')
             .conv(5, 5, 128, 1, 1, name='icp8_out2'))

        (self.feed('icp8_in')
             .max_pool(3, 3, 1, 1, name='icp8_pool')
             .conv(1, 1, 128, 1, 1, name='icp8_out3'))

        (self.feed('icp8_in')
             .conv(1, 1, 256, 1, 1, name='icp8_out0'))

        (self.feed('icp8_out0',
                   'icp8_out1',
                   'icp8_out2',
                   'icp8_out3')
             .concat(3, name='icp8_out')
             .conv(1, 1, 192, 1, 1, name='icp9_reduction1')
             .conv(3, 3, 384, 1, 1, name='icp9_out1'))

        (self.feed('icp8_out')
             .conv(1, 1, 48, 1, 1, name='icp9_reduction2')
             .conv(5, 5, 128, 1, 1, name='icp9_out2'))

        (self.feed('icp8_out')
             .max_pool(3, 3, 1, 1, name='icp9_pool')
             .conv(1, 1, 128, 1, 1, name='icp9_out3'))

        (self.feed('icp8_out')
             .conv(1, 1, 384, 1, 1, name='icp9_out0'))

        (self.feed('icp9_out0',
                   'icp9_out1',
                   'icp9_out2',
                   'icp9_out3')
             .concat(3, name='icp9_out')
             .Seblock(1024,name='se')
             #.dwise_conv(k_h=7, k_w=7, channel_multiplier=1, strides=[1, 1, 1, 1], padding='SAME', stddev=0.002, name='dwise_conv3')
             .conv(7, 7, 1024, 1, 1, relu=True, name='dw3')
             #.conv(1, 1, 1024, 1, 1,relu=False, name='pw3')
             .fc(2048, name='cls3_fc1_pose')
             .fc(3, relu=False, name='cls3_fc_pose_xyz'))

