# Main and efficient version of the bottleneck block (the other one is mnblock_sep.py)

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
# from utils import ChannelShuffle

def bn_layer(out_channels, kernel_size=1, stride=1, padding="valid", groups=1, bias=False, act=layers.Activation("silu")):
    return tf.keras.Sequential([
        layers.Conv2D(out_channels, kernel_size, stride, padding, groups=groups, use_bias=bias),
        layers.BatchNormalization(),
        # ChannelShuffle(out_channels//2),
        act,
    ])

class SEBlock(layers.Layer):
    def __init__(self, channels, groups=1, se_ratio=0.25, act=layers.Activation('silu')):
        super(SEBlock, self).__init__()

        reduced_channels = max(1, int(channels * se_ratio))
        
        self.se = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(keepdims=True),
            layers.Conv2D(reduced_channels, kernel_size=1, groups=groups),
            act,
            layers.Conv2D(channels, kernel_size=1, groups=groups),
        ])

    def call(self, x):
        return x * activations.sigmoid(self.se(x))

class InvertedBottleneck(tf.keras.Model):
    def __init__(self, in_channels, out_channels, expansion_factor=2, split_input=True, split_factor=32, fused=False, kernel=3, stride=1, padding="same", use_se=True, se_ratio=0.25):
        super(InvertedBottleneck, self).__init__()

        mid_channels = in_channels * expansion_factor
        
        split_size = min(in_channels, split_factor) if split_input else in_channels
        assert in_channels % split_size == 0

        self.num_branches = in_channels // split_size # can be 1

        self.bottleneck = tf.keras.Sequential()
        
        if fused:
            # Regular convolution [GROUPED]
            self.bottleneck.add(bn_layer(mid_channels, kernel_size=kernel, stride=stride, padding=padding, groups=self.num_branches))
        else:
            # Pointwise (1x1) expansion [GROUPED]
            self.bottleneck.add(bn_layer(mid_channels, groups=self.num_branches))
            # Depthwise convolution
            self.bottleneck.add(bn_layer(mid_channels, kernel_size=kernel, stride=stride, padding=padding, groups=mid_channels))
        
        if self.num_branches > 1 and use_se:
            # Squeeze and Excitation mechanism [GROUPED]
            self.bottleneck.add(SEBlock(mid_channels, groups=self.num_branches, se_ratio=se_ratio))
            # Pointwise (1x1) linear projection [GROUPED]
            self.bottleneck.add(bn_layer(mid_channels, groups=self.num_branches, act=layers.Activation('linear')))

        if use_se:
            # Squeeze and Excitation mechanism
            self.bottleneck.add(SEBlock(mid_channels, se_ratio=se_ratio))

        # Pointwise (1x1) linear projection
        self.bottleneck.add(bn_layer(out_channels, act=layers.Activation('linear')))

    def call(self, x):
        y = self.bottleneck(x)
        return y + x if y.shape == x.shape else y


if __name__ == "__main__":
    bottleneck = InvertedBottleneck(1, 32)
    model = tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(128, 1000, 1)), bottleneck])
    model.summary()