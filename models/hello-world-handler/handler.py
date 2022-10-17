class EndpointHandler():
    def __init__(self, path=""):
        pass
    def __call__(self, data):
        """
        Args:
            data (:obj:):
                includes the input data and the parameters for the inference.
        Return:
            A :obj:`dict`:. base64 encoded image
        """
        inputs = data.pop("inputs", data)
        return [f"Hello, {x}" for x in inputs]