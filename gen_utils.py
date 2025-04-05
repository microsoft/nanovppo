
class GeneratorClient:
    backend = "sgl"

    @classmethod
    def init(cls, backend, **kwargs):
        cls.backend = backend
        if cls.backend == "sgl":
            from sgl_utils import SGLGeneratorClient
            SGLGeneratorClient(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {cls.backend}")

    @classmethod
    def get(cls):
        if cls.backend == "sgl":
            from sgl_utils import SGLGeneratorClient
            return SGLGeneratorClient.get()
        else:
            raise ValueError(f"Unsupported backend: {cls.backend}")
