import tensorflow as tf

deserialize_keras_object = tf.keras.utils.deserialize_keras_object
serialize_keras_object = tf.keras.utils.serialize_keras_object

class QuantizeAnnotate(tf.keras.layers.Wrapper):
    def __init__(self, layer, quantize_config=None, **kwargs):
        super(QuantizeAnnotate, self).__init__(layer, **kwargs)

        self.quantize_config = quantize_config

        if (not hasattr(self, '_batch_input_shape') and hasattr(layer, '_batch_input_shape')):
            self._batch_input_shape = self.layer._batch_input_shape
    
    def call(self, inputs, training=None):
        return self.layer.call(inputs)
    
    def get_config(self):
        base_config = super(QuantizeAnnotate, self).get_config()
        config = {'quantize_config': serialize_keras_object(self.quantize_config)}
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config):
        config = config.copy()

        quantize_config = deserialize_keras_object(
            config.pop('quantize_config'),
            module_objects=None,
            custom_objects=None)

        layer = tf.keras.layers.deserialize(config.pop('layer'))
        return cls(layer=layer, quantize_config=quantize_config, **config)
    
    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
    
    @property
    def trainable(self):
        return self.layer.trainable
    
    @trainable.setter
    def trainable(self, value):
        self.layer.trainable = value
    
    @property
    def trainable_weights(self):
        return self.layer.trainable_weights
    
    @property
    def non_trainable_weights(self):
        return self.layer.non_trainable_weights
    
    @property
    def updates(self):
        return self.layer.updates
    
    @property
    def losses(self):
        return self.layer.losses
    
    def get_weights(self):
        return self.layer.get_weights()
    
    def set_weights(self, weights):
        self.layer.set_weights(weights)