import tensorflow as tf

class EfficientNet(tf.keras.Model):
    def __init__(self, num_classes=200, pretrained=False):
        super(EfficientNet, self).__init__()

        self.pretrained = pretrained

        self.model = tf.keras.applications.EfficientNetB2(
            include_top=False,
            weights='imagenet' if pretrained else None,
            input_shape=(128, 1000, 1)
        )

        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()

        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        x = self.model(inputs)
        x = self.global_avg_pooling(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = EfficientNet()
    model.build((None, 128, 1000, 1))
    model.summary()