# Another version of mnblock.py that does not exploit "groups" arg in Conv2d and instead uses for-cycles, lists, and concatenations
# They have the same number of parameters and behave the same given the same input
# but this version performs worse most likely because kernels of each branch are initialized independently

# We can say that this version is wrong because the input and its weight matrix should be treated always in a standalone fashion
# but for the moment in which they may have to be split over the channels dimension to perform independent (grouped) convolutions

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations

def bn_layer(out_channels, kernel_size=1, stride=1, padding="valid", groups=1, bias=False, act=layers.Activation("silu")):
    return tf.keras.Sequential([
        layers.Conv2D(out_channels, kernel_size, stride, padding, groups=groups, use_bias=bias),
        layers.BatchNormalization(),
        act,
    ])

class SEBlock(layers.Layer):
    def __init__(self, channels, se_ratio=0.25, act=layers.Activation('silu')):
        super(SEBlock, self).__init__()

        reduced_channels = max(1, int(channels * se_ratio))
        
        self.se = tf.keras.Sequential([
            layers.GlobalAveragePooling2D(keepdims=True),
            layers.Conv2D(reduced_channels, kernel_size=1),
            act,
            layers.Conv2D(channels, kernel_size=1),
        ])

    def call(self, x):
        return x * activations.sigmoid(self.se(x))

class InvertedBottleneck(tf.keras.Model):
    def __init__(self, in_channels, out_channels, expansion_factor=2, split_input=True, split_factor=32, fused=False, kernel=3, stride=1, padding="same", use_se=True, se_ratio=0.25):
        super(InvertedBottleneck, self).__init__()

        mid_channels = in_channels * expansion_factor
        
        split_size = min(in_channels, split_factor) if split_input else in_channels
        assert in_channels % split_size == 0 # can be 1

        self.num_branches = in_channels // split_size
        # branch_in_channels = split_size
        branch_out_channels = mid_channels // self.num_branches

        self.branches = [self.build_branch(branch_out_channels, fused, kernel, stride, padding, use_se, se_ratio) for _ in range(self.num_branches)]

        self.final = self.build_final(mid_channels, out_channels, use_se, se_ratio)

    def build_branch(self, out_channels, fused, kernel, stride, padding, use_se, se_ratio):
        branch_modules = tf.keras.Sequential()

        if fused:
            # Regular convolution
            branch_modules.add(bn_layer(out_channels, kernel_size=kernel, stride=stride, padding=padding))
        else:
            # Pointwise (1x1) expansion
            branch_modules.add(bn_layer(out_channels))
            # Depthwise convolution
            branch_modules.add(bn_layer(out_channels, kernel_size=kernel, stride=stride, padding=padding, groups=out_channels))

        if self.num_branches > 1 and use_se:
            # Squeeze and Excitation mechanism
            branch_modules.add(SEBlock(out_channels, se_ratio=se_ratio))
            # Pointwise (1x1) linear projection
            branch_modules.add(bn_layer(out_channels, act=layers.Activation('linear')))

        return branch_modules

    def build_final(self, in_channels, out_channels, use_se, se_ratio):
        final_modules = tf.keras.Sequential()

        if use_se:
            # Squeeze and Excitation mechanism
            final_modules.add(SEBlock(in_channels, se_ratio=se_ratio))

        # Pointwise (1x1) linear projection
        final_modules.add(bn_layer(out_channels, act=layers.Activation('linear')))

        return final_modules

    def call(self, x):
        chunks = tf.split(x, self.num_branches, axis=3)
        y = tf.concat([branch(chunk) for chunk, branch in zip(chunks, self.branches)], axis=3)
        y = self.final(y)
        return y + x if y.shape == x.shape else y


if __name__ == "__main__":
    bottleneck = InvertedBottleneck(1, 32)
    model = tf.keras.models.Sequential([layers.InputLayer(input_shape=(128, 1000, 1)), bottleneck])
    model.summary()