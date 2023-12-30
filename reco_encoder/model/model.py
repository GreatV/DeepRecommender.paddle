import paddle


def activation(input, kind):
    if kind == "selu":
        return paddle.nn.functional.selu(x=input)
    elif kind == "relu":
        return paddle.nn.functional.relu(x=input)
    elif kind == "relu6":
        return paddle.nn.functional.relu6(x=input)
    elif kind == "sigmoid":
        return paddle.nn.functional.sigmoid(x=input)
    elif kind == "tanh":
        return paddle.nn.functional.tanh(x=input)
    elif kind == "elu":
        return paddle.nn.functional.elu(x=input)
    elif kind == "lrelu":
        return paddle.nn.functional.leaky_relu(x=input)
    elif kind == "swish":
        return input * paddle.nn.functional.sigmoid(x=input)
    elif kind == "none":
        return input
    else:
        raise ValueError("Unknown non-linearity type")


def MSEloss(inputs, targets, size_average=False):
    mask = targets != 0
    num_ratings = paddle.sum(x=mask.astype(dtype="float32"))
    criterion = paddle.nn.MSELoss(reduction="sum" if not size_average else "mean")
    return criterion(inputs * mask.astype(dtype="float32"), targets), paddle.to_tensor(
        data=[1.0], dtype="float32"
    ) if size_average else num_ratings


class AutoEncoder(paddle.nn.Layer):
    def __init__(
        self,
        layer_sizes,
        nl_type="selu",
        is_constrained=True,
        dp_drop_prob=0.0,
        last_layer_activations=True,
    ):
        """
        Describes an AutoEncoder model
        :param layer_sizes: Encoder network description. Should start with feature size (e.g. dimensionality of x).
        For example: [10000, 1024, 512] will result in:
          - encoder 2 layers: 10000x1024 and 1024x512. Representation layer (z) will be 512
          - decoder 2 layers: 512x1024 and 1024x10000.
        :param nl_type: (default 'selu') Type of no-linearity
        :param is_constrained: (default: True) Should constrain decoder weights
        :param dp_drop_prob: (default: 0.0) Dropout drop probability
        :param last_layer_activations: (default: True) Whether to apply activations on last decoder layer
        """
        super(AutoEncoder, self).__init__()
        self._dp_drop_prob = dp_drop_prob
        self._last_layer_activations = last_layer_activations
        if dp_drop_prob > 0:
            self.drop = paddle.nn.Dropout(p=dp_drop_prob)
        self._last = len(layer_sizes) - 2
        self._nl_type = nl_type
        self.encode_w = paddle.nn.ParameterList(
            parameters=[
                paddle.create_parameter(
                    shape=[layer_sizes[i + 1], layer_sizes[i]],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Uniform(),
                )
                for i in range(len(layer_sizes) - 1)
            ]
        )
        for ind, w in enumerate(self.encode_w):
            init_XavierUniform = paddle.nn.initializer.XavierUniform()
            init_XavierUniform(w)
        self.encode_b = paddle.nn.ParameterList(
            parameters=[
                paddle.create_parameter(
                    shape=[layer_sizes[i + 1]],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(value=0.0),
                )
                for i in range(len(layer_sizes) - 1)
            ]
        )
        reversed_enc_layers = list(reversed(layer_sizes))
        self.is_constrained = is_constrained
        if not is_constrained:
            self.decode_w = paddle.nn.ParameterList(
                parameters=[
                    paddle.create_parameter(
                        shape=[reversed_enc_layers[i + 1], reversed_enc_layers[i]],
                        dtype="float32",
                        default_initializer=paddle.nn.initializer.Uniform(),
                    )
                    for i in range(len(reversed_enc_layers) - 1)
                ]
            )
            for ind, w in enumerate(self.decode_w):
                init_xavier_uniform = paddle.nn.initializer.XavierUniform()
                init_xavier_uniform(w)
        self.decode_b = paddle.nn.ParameterList(
            parameters=[
                paddle.create_parameter(
                    shape=[reversed_enc_layers[i + 1]],
                    dtype="float32",
                    default_initializer=paddle.nn.initializer.Constant(value=0.0),
                )
                for i in range(len(reversed_enc_layers) - 1)
            ]
        )
        print("******************************")
        print("******************************")
        print(layer_sizes)
        print("Dropout drop probability: {}".format(self._dp_drop_prob))
        print("Encoder pass:")
        for ind, w in enumerate(self.encode_w):
            print(w.data.shape)
            print(self.encode_b[ind].shape)
        print("Decoder pass:")
        if self.is_constrained:
            print("Decoder is constrained")
            for ind, w in enumerate(list(reversed(self.encode_w))):
                x = w
                perm_0 = list(range(x.ndim))
                perm_0[0] = 1
                perm_0[1] = 0
                print(x.transpose(perm=perm_0).shape)
                print(self.decode_b[ind].shape)
        else:
            for ind, w in enumerate(self.decode_w):
                print(w.data.shape)
                print(self.decode_b[ind].shape)
        print("******************************")
        print("******************************")

    def encode(self, x):
        for ind, w in enumerate(self.encode_w):
            x = activation(
                input=paddle.nn.functional.linear(
                    weight=w.T, bias=self.encode_b[ind], x=x
                ),
                kind=self._nl_type,
            )
        if self._dp_drop_prob > 0:
            x = self.drop(x)
        return x

    def decode(self, z):
        if self.is_constrained:
            for ind, w in enumerate(list(reversed(self.encode_w))):
                x = w
                perm_1 = list(range(x.ndim))
                perm_1[0] = 1
                perm_1[1] = 0
                z = activation(
                    input=paddle.nn.functional.linear(
                        weight=x.transpose(perm=perm_1).T, bias=self.decode_b[ind], x=z
                    ),
                    kind=self._nl_type
                    if ind != self._last or self._last_layer_activations
                    else "none",
                )
        else:
            for ind, w in enumerate(self.decode_w):
                z = activation(
                    input=paddle.nn.functional.linear(
                        weight=w.T, bias=self.decode_b[ind], x=z
                    ),
                    kind=self._nl_type
                    if ind != self._last or self._last_layer_activations
                    else "none",
                )
        return z

    def forward(self, x):
        return self.decode(self.encode(x))
