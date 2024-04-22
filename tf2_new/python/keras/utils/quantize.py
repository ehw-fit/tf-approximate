import tensorflow as tf

from python.keras.utils import (quantize_wrapper, quantize_annotate)
from python.keras.utils.approx_quantize_config import (
    QuantConv2DConfig, QuantDepthwiseConv2DConfig,
    TunableQuantConv2DConfig, TunableQuantDepthwiseConv2DConfig)


def quantize_annotate_layer(to_annotate, quantize_config=None):
    return quantize_annotate.QuantizeAnnotate(layer=to_annotate, quantize_config=quantize_config)


def quantize_annotate_model(model, layer_type_quantize_map=None, layer_override_quantize_map=None):
    def _annotate_approx_layer(layer, quantize_map, quantize_override_map):
        if quantize_override_map is not None and layer.name in quantize_override_map:
            return quantize_annotate_layer(layer, quantize_config=quantize_override_map[layer.name])
        elif type(layer) in quantize_map:
            create_layer_quantize_config = quantize_map[type(layer)]
            return quantize_annotate_layer(layer, quantize_config=create_layer_quantize_config())
        else:
            return layer
    
    if layer_type_quantize_map is None:
        layer_type_quantize_map = {
        tf.keras.layers.Conv2D: lambda: TunableQuantConv2DConfig(8, ''),
        tf.keras.layers.DepthwiseConv2D: lambda: TunableQuantDepthwiseConv2DConfig(8, '')
    }
    
    return tf.keras.models.clone_model(model, input_tensors=None, 
        clone_function=lambda layer: _annotate_approx_layer(layer, layer_type_quantize_map, 
                                                            layer_override_quantize_map))


def quantize_apply(model):
    def _clone_model_with_weights(model_to_clone):
        cloned_model = tf.keras.models.clone_model(model_to_clone)
        cloned_model.set_weights(model_to_clone.get_weights())

        return cloned_model

    def _extract_original_model(model_to_unwrap):
        layer_quantize_map = {}

        def _unwrap(layer):
            if not isinstance(layer, quantize_annotate.QuantizeAnnotate):
                return layer
            
            annotate_wrapper = layer
            layer_quantize_map[annotate_wrapper.layer.name] = {
                'quantize_config': annotate_wrapper.quantize_config
            }
            return annotate_wrapper.layer
        
        unwrapped_model = tf.keras.models.clone_model(model_to_unwrap, input_tensors=None, clone_function=_unwrap)
        return unwrapped_model, layer_quantize_map

    def _quantize(layer, layer_quantize_map):
        if layer.name not in layer_quantize_map:
            return layer
        
        quantize_config = layer_quantize_map[layer.name].get('quantize_config')
        return quantize_wrapper.QuantizeWrapper(layer, quantize_config)
    
    model_copy = _clone_model_with_weights(model)
    unwrapped_model, layer_quantize_map = _extract_original_model(model_copy)

    return tf.keras.models.clone_model(unwrapped_model, input_tensors=None, clone_function=lambda layer: _quantize(layer, layer_quantize_map))

def fake_quantize_apply(model):
    return model