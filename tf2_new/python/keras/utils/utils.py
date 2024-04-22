from tensorflow.python.framework import smart_cond as smart_module
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables

def smart_cond(pred, true_fn=None, false_fn=None, name=None):
    if isinstance(pred, variables.Variable):
        return control_flow_ops.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)
    
    return smart_module.smart_cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)