import paddle
from reco_encoder.model import model

if __name__ == "__main__":
    encoder = model.AutoEncoder(layer_sizes=[19547, 256, 128], is_constrained=False)
    x = paddle.randn(shape=[64, 19547])
    try:
        x = paddle.static.InputSpec.from_tensor(x)
        paddle.jit.save(encoder, input_spec=(x,), path="./model")
        print("[JIT] paddle.jit.save successed.")
        exit(0)
    except Exception as e:
        print("[JIT] paddle.jit.save failed.")
        raise e
