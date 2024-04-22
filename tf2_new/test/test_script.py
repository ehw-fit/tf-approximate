import json

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import sys
sys.path.append("../")
from python.keras.layers.fake_convolutional import FakeApproxConv2D, FakeApproxDepthwiseConv2D
import tensorflow.python.keras.layers as tf_layers
from tensorflow.keras import backend as K
import argparse


def create_model():
    # add input layers
    input_tensor = tf.keras.Input(shape=(32, 32, 3))    # cifar-10 dataset
    resized_input = tf.keras.layers.experimental.preprocessing.Resizing(
        128, 128)(input_tensor)

    # add base model
    base_model = ResNet50V2(input_shape=(128, 128, 3),
                            input_tensor=resized_input,
                            include_top=False,
                            weights="imagenet",
                            pooling="avg")

    # add custom top layers
    flatten = tf_layers.Flatten()(base_model.output)
    bn1 = tf_layers.BatchNormalization()(flatten)
    dense1 = tf_layers.Dense(128, "relu")(bn1)
    drop1 = tf_layers.Dropout(0.3)(dense1)
    bn2 = tf_layers.BatchNormalization()(drop1)
    dense2 = tf_layers.Dense(64, "relu")(bn2)
    drop2 = tf_layers.Dropout(0.3)(dense2)
    dense2 = tf_layers.Dense(10, "softmax")(drop2)

    # construct keras model
    resnet_model = tf.keras.Model(
        inputs=base_model.layers[0].output, outputs=dense2)

    with tf.device("/gpu:0"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Convert class vectors to binary class matrices (one-hot vectors)
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        # preprocess
        x_train = tf.keras.applications.resnet_v2.preprocess_input(x_train)
        x_test = tf.keras.applications.resnet_v2.preprocess_input(x_test)

        # set model to train
        resnet_model.trainable = True
        for l in resnet_model.layers:
            l.trainable = True

        # compile model
        resnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        # train
        resnet_model.fit(x=x_train, y=y_train, batch_size=64, epochs=3)

        # save model and weights to separate files
        resnet_model.save("./resnet50V2.h5")
        resnet_model.save_weights("./resnet50V2_weights.h5")


def get_indices_of_layers_of_type(layers, layer_type):
    indices = []
    for i in range(len(layers)):
        if layers[i].__class__.__name__ == layer_type:
            indices.append(i)
    return indices


def get_interconnection_graph(layers):
    def _find_index_of(layer, tmp_layers):
        for i in range(len(tmp_layers)):
            if tmp_layers[i] is layer:
                return i
        return -1

    graph = [[] for l in layers]
    for i in range(len(layers)):
        in_layers = layers[i]._inbound_nodes[0].inbound_layers
        if isinstance(in_layers, list):
            in_layers = [_find_index_of(x, layers) for x in in_layers]
            graph[i] = in_layers
        else:
            graph[i] = _find_index_of(in_layers, layers)
    return graph


def substitute_layers(layers, list_of_layers_to_substitute, list_of_mults, mults_bin_path,
    per_channel):
    conf_args = ["kernel_size", "filters", "strides", "padding", "data_format",
                 "use_bias", "activation"]
    for l, m in zip(list_of_layers_to_substitute, list_of_mults):
        layer_conf = layers[l].get_config()
        approx_mul_file = mults_bin_path + m + ".bin"
        params = {}

        for arg in conf_args:
            if arg in layer_conf.keys():
                params[arg] = layer_conf[arg]
        if layers[l].__class__.__name__ == "Conv2D":
            newL = FakeApproxConv2D(
                **params,
                approx_mul_table_file=approx_mul_file,
                per_channel=per_channel)
        else:
            raise ValueError("Bad layer type to substitute {}!".format(
                layers[l].__class__.__name__))
        layers[l] = newL
    return layers


def construct_new_approx_model(layers_to_substitute, mult_list, perchannel):

    # load (accurate) base model (better than construct from scratch)
    model = tf.keras.models.load_model("./resnet50V2.h5")
    model_layers = [l for l in model.layers]

    _interconnection_graph = get_interconnection_graph(model_layers)
    _mults_bin_dir = "../examples/axmul_8x8/"

    substitute_layers(model_layers, layers_to_substitute,
                      mult_list, _mults_bin_dir, perchannel)

    # reconstruct the model from the list of layers and interconnection graph
    for i in range(1, len(model_layers)):
        # skip first layer (has no input layers)
        if isinstance(_interconnection_graph[i], list):
            # layer has more input layers
            inputs = [model_layers[j] for j in _interconnection_graph[i]]
        else:
            inputs = model_layers[_interconnection_graph[i]]
        if isinstance(inputs, tf.keras.layers.InputLayer):
            # special case if input layer is of type InputLayer
            inputs = inputs.output
        model_layers[i] = model_layers[i](inputs)

    out_approx_model = tf.keras.Model(
        inputs=model_layers[0].output, outputs=model_layers[-1])

    return out_approx_model

def describe_layer(layer):
    r = {}
    r["filters"] = layer.filters
    r["input_shape"] = layer.input.shape
    r["kernel_size"] = layer.kernel_size
    r["strides"] = layer.strides
    r["padding"] = layer.padding
    r["use_bias"] = layer.use_bias

    return "\n".join(f" - {i}: {v}" for i, v in r.items())

if __name__ == "__main__":
    # list of indices of Conv2D layers in resnet50V2 model
    # can be obtained by calling get_indices_of_layers_of_type(resnet_model_layers, "Conv2D")
    conv_layers = [3, 8, 12, 15, 16, 20, 24, 27, 31, 35, 39, 43, 47, 50, 51, 55, 59, 62, 66, 70, 73, 77, 81, 85, 89, 93, 96, 97, 101, 105,
                108, 112, 116, 119, 123, 127, 130, 134, 138, 141, 145, 149, 153, 157, 161, 164, 165, 169, 173, 176, 180, 184, 187]

    parser = argparse.ArgumentParser()
    parser.add_argument('-create', action='store_true')
    args = parser.parse_args()

    if args.create:
        create_model()
        exit(0)

    perchannel = False
    type = "perchannel" if perchannel else "perbatch"
    data =  [] #json.load(open("tmp/data.json"))
    batch = 128
    dump = False



    with tf.device("/gpu:0"):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        x_test = x_test[:3*batch]
        y_test = y_test[:3*batch]
        # Convert class vectors to binary class matrices (one-hot vectors)
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)

        # preprocess
        x_train = tf.keras.applications.resnet_v2.preprocess_input(x_train)
        x_test = tf.keras.applications.resnet_v2.preprocess_input(x_test)

        ax_layers = [conv_layers] + [conv_layers[1:]] + [[i] for i in conv_layers]

        exists = [i["layers"]
                for i in data if i["type"] == type and i["batch"] == batch]
        ax_layers = [i for i in ax_layers if i not in exists]

        ax_layers = [[i] for i in conv_layers]
        for i, layers in enumerate(ax_layers):

            # create model with approximated layers 8 and 15 with mults 1JFF
            model = construct_new_approx_model(
                layers, ["mul8u_1JFF"] * len(layers), perchannel)
            model.trainable = False
            for l in model.layers:
                l.trainable = False
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            model.load_weights("./resnet50V2_weights.h5")
            model.summary()

            if dump and len(layers) == 1:
                l = layers[0]
                print(dir(model.layers[l]))
                print(model.layers[l].filters)
                fn = K.function([model.input], [model.layers[l].input, model.layers[l].output])

                for b in [0, 1]:
                    res = {}
                    res["input"], res["output"] = fn([x_test[(b) * 128: (b+1) * 128]])
                    for i, j in enumerate(model.layers[l].weights):
                        res[f"weights_{i}"] = j
                    res[f"bias"] = model.layers[l].bias

                    res[f"filters"] = model.layers[l].filters
                    res[f"kernel_size"] = model.layers[l].kernel_size
                    
                    np.savez_compressed(f"tmp/layer_{l}_batch_{b}.npz", 
                        **res)

            else:

                for l in conv_layers:
                    print("# layer: ", l)
                    print(describe_layer(model.layers[l]))
                    
                    print("="*80)


#              exit()


                loss, acc = model.evaluate(x=x_test, y=y_test, batch_size=128)
                data.append({
                    "type": type,
                    "layers": layers,
                    "layers_cnt": len(layers),
                    "accuracy": acc,
                    "loss": loss,
                    "batch": 128
                })
                print(f"{i} / {len(ax_layers)} Accuracy: {acc} {layers}")
                json.dump(data, open(f"tmp/data2.json", "w"))
    # model.summary()
