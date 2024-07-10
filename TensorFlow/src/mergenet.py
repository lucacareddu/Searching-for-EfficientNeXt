import tensorflow as tf
from mnblock import InvertedBottleneck as mnblock
# from mnblock_sep import InvertedBottleneck as mnblock

class MergeNet(tf.keras.Model):
    def __init__(self, num_classes=200, split=True):
        super(MergeNet, self).__init__()

        self.split = split

        self.conv_stem = tf.keras.Sequential([
            mnblock(1, 32, expansion_factor=32, fused=True, stride=2, padding="same", split_input=split),
            mnblock(32, 128, expansion_factor=4, fused=True, stride=2, padding="same", split_input=split),
        ])
        
        self.bottlenecks = tf.keras.Sequential([
            mnblock(128, 128, expansion_factor=2, split_input=split),
            mnblock(128, 256, expansion_factor=2, stride=2, padding="same", split_input=split),
            mnblock(256, 256, expansion_factor=2, split_input=split),
            mnblock(256, 512, expansion_factor=2, stride=2, padding="same", split_input=split),
            mnblock(512, 512, expansion_factor=2, split_input=split),
            mnblock(512, 1024, expansion_factor=2, stride=2, padding="same", split_input=split),
            mnblock(1024, 1024, expansion_factor=2, split_input=split),
        ])
        
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.conv_stem(x)
        x = self.bottlenecks(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = MergeNet()
    model.build((None, 128, 1000, 1))
    model.summary()