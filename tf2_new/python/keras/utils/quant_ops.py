import tensorflow as tf

from tensorflow.python.training import moving_averages

def assign(ref, value, name=None):
    if hasattr(tf, 'assign'):
        return tf.assign(ref, value, name=name)
    else:
        return ref.assign(value, name=name)


def FixedQuantize(inputs, init_min=-6.0, init_max=6.0, scope=None):
    if scope is None:
        scope = 'FixedQuantize'
    
    with tf.name_scope(scope):
        return tf.quantization.fake_quant_with_min_max_args(
            inputs, min=init_min, max=init_max)


def FakeQuantize(inputs, 
                 min_var, 
                 max_var, 
                 name_prefix='FakeQuantize', 
                 num_bits=8, 
                 narrow_range=False, 
                 symmetric=False):
    with tf.name_scope(name_prefix):
        batch_min = tf.math.reduce_min(inputs, name='BatchMin')
        batch_max = tf.math.reduce_max(inputs, name='BatchMax')

        if symmetric:
            if narrow_range:
                min_max_ratio = -1
            else:
                min_max_ratio = -((1 << num_bits) - 2) / (1 << num_bits)
            
            batch_min = tf.math.minimum(batch_min, batch_max / min_max_ratio)
            batch_max = tf.math.maximum(batch_max, batch_min * min_max_ratio)
        
        # range_min = tf.math.minimum(tf.math.minimum(min_var, batch_min), 0.0)
        # range_max = tf.math.maximum(tf.math.maximum(max_var, batch_max), 0.0)

        assign_min = assign(min_var, batch_min, name='AssignMinFake')
        assign_max = assign(max_var, batch_max, name='AssignMaxFake')
        
        return _FakeQuantWithMinMaxVars(
            inputs,
            assign_min,
            assign_max,
            per_channel=False,
            num_bits=num_bits,
            narrow_range=narrow_range)


def AllValuesQuantize(inputs,
                      min_var,
                      max_var,
                      name_prefix='AllValuesQuantize',
                      is_training=True,
                      num_bits=8,
                      narrow_range=False,
                      symmetric=False):
    with tf.name_scope(name_prefix):
        if not is_training:
            return _FakeQuantWithMinMaxVars(
                inputs,
                min_var,
                max_var,
                per_channel=False,
                num_bits=num_bits,
                narrow_range=narrow_range)
            
        batch_min = tf.math.reduce_min(inputs, name='BatchMin')
        batch_max = tf.math.reduce_max(inputs, name='BatchMax')

        if symmetric:
            if narrow_range:
                min_max_ratio = -1
            else:
                min_max_ratio = -((1 << num_bits) - 2) / (1 << num_bits)
            
            batch_min = tf.math.minimum(batch_min, batch_max / min_max_ratio)
            batch_max = tf.math.maximum(batch_max, batch_min * min_max_ratio)
        
        range_min = tf.math.minimum(tf.math.minimum(min_var, batch_min), 0.0)
        range_max = tf.math.maximum(tf.math.maximum(max_var, batch_max), 0.0)

        assign_min = assign(min_var, range_min, name='AssignMinAllValue')
        assign_max = assign(max_var, range_max, name='AssignMaxAllValue')

        return _FakeQuantWithMinMaxVars(
            inputs,
            assign_min,
            assign_max,
            per_channel=False,
            num_bits=num_bits,
            narrow_range=narrow_range)


def LastValueQuantize(inputs,
                      min_var,
                      max_var,
                      per_channel=False,
                      name_prefix='LastValueQuant',
                      is_training=True,
                      num_bits=8,
                      narrow_range=False,
                      symmetric=False):
    with tf.name_scope(name_prefix):
        input_shape = inputs.get_shape()
        input_dim   = len(input_shape)

        if not is_training:
            return _FakeQuantWithMinMaxVars(
                inputs,
                min_var,
                max_var,
                per_channel=per_channel,
                num_bits=num_bits,
                narrow_range=narrow_range)
        
        if per_channel:
            if input_dim == 2:
                reduce_dims = [0]
            elif input_dim == 4:
                reduce_dims = [0, 1, 2]
        
        if per_channel:
            if input_dim >= 2:
                batch_min = tf.math.reduce_min(
                    inputs, axis=reduce_dims, name='BatchMin')
            else:
                batch_min = inputs
        else:
            batch_min = tf.math.reduce_min(inputs, name='BatchMin')
        
        if per_channel:
            if input_dim >= 2:
                batch_max = tf.math.reduce_max(
                    inputs, axis=reduce_dims, name='BatchMax')
            else:
                batch_max = inputs
        else:
            batch_max = tf.math.reduce_max(inputs, name='BatchMax')
        
        if symmetric:
            if narrow_range:
                min_max_ratio = -1
            else:
                min_max_ratio = -((1 << num_bits) - 2) / (1 << num_bits)
            
            range_min = tf.math.minimum(batch_min, batch_max / min_max_ratio)
            range_max = tf.math.maximum(batch_max, batch_min * min_max_ratio)
        else:
            range_min = tf.math.minimum(batch_min, 0)
            range_max = tf.math.maximum(batch_max, 0)
        
        assign_min = assign(min_var, range_min, name='AssignMinLast')
        assign_max = assign(max_var, range_max, name='AssignMaxLast')

        return _FakeQuantWithMinMaxVars(
            inputs,
            assign_min,
            assign_max,
            per_channel=per_channel,
            num_bits=num_bits,
            narrow_range=narrow_range)


def MovingAvgQuantize(inputs,
                      min_var,
                      max_var,
                      per_channel=False,
                      ema_decay=0.999,
                      name_prefix='MovingAvgQuantize',
                      is_training=True,
                      num_bits=8,
                      narrow_range=False,
                      symmetric=False):
    with tf.name_scope(name_prefix):
        input_shape = inputs.get_shape()
        input_dim   = len(input_shape)

        if not is_training:
            return _FakeQuantWithMinMaxVars(
                inputs,
                min_var,
                max_var,
                per_channel=per_channel,
                num_bits=num_bits,
                narrow_range=narrow_range)
        
        if per_channel:
            if input_dim == 2:
                reduce_dims = [0]
            elif input_dim == 4:
                reduce_dims = [0, 1, 2]
        
        if per_channel:
            if input_dim >= 2:
                batch_min = tf.math.reduce_min(
                    inputs, axis=reduce_dims, name='BatchMin')
            else:
                batch_min = inputs
        else:
            batch_min = tf.math.reduce_min(inputs, name='BatchMin')
        
        if per_channel:
            if input_dim >= 2:
                batch_max = tf.math.reduce_max(
                    inputs, asix=reduce_dims, name='BatchMax')
            else:
                batch_max = inputs
        else:
            batch_max = tf.math.reduce_max(inputs, name='BatchMax')
        
        if symmetric:
            if narrow_range:
                min_max_ratio = -1
            else:
                min_max_ratio = -((1 << num_bits) - 2) / (1 << num_bits)
            
            range_min = tf.minimum(batch_min, batch_max / min_max_ratio)
            range_max = tf.maximum(batch_max, batch_min * min_max_ratio)
        else:
            range_min = tf.minimum(batch_min, 0.0)
            range_max = tf.maximum(batch_max, 0.0)
        
        assign_min = moving_averages.assign_moving_average(
            min_var, range_min, ema_decay, zero_debias=False, name='AssignMinEma')
        assign_max = moving_averages.assign_moving_average(
            max_var, range_max, ema_decay, zero_debias=False, name='AssignMaxEma')
        
        return _FakeQuantWithMinMaxVars(
            inputs,
            assign_min,
            assign_max,
            per_channel=per_channel,
            num_bits=num_bits,
            narrow_range=narrow_range)


def _FakeQuantWithMinMaxVars(inputs, min_var, max_var, per_channel, num_bits, 
                             narrow_range):
    if per_channel:
        assert len(min_var.get_shape()) == 1
        assert len(max_var.get_shape()) == 1
        return tf.quantization.fake_quant_with_min_max_vars_per_channel(
            inputs, min_var, max_var, num_bits=num_bits, narrow_range=narrow_range)
    else:
        assert min_var.get_shape() == []
        assert max_var.get_shape() == []
        return tf.quantization.fake_quant_with_min_max_vars(
            inputs, min_var, max_var, num_bits=num_bits, narrow_range=narrow_range)
