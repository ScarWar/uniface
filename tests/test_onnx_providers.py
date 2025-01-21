from uniface import retinaface


def test_default_provider():
    provider = retinaface.OnnxProvider()

    assert provider.to_list() == ["CPUExecutionProvider", {}]


def test_cuda_provider():
    provider = retinaface.OnnxProvider("cuda", 2)

    assert provider.to_list() == ["CUDAExecutionProvider", {"device_id": 2}]
