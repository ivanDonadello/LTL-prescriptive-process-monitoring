class DTInput(object):

    def __init__(self, prefix_length, features, encoded_data, labels):
        self.prefix_length = prefix_length
        self.features = features
        self.encoded_data = encoded_data
        self.labels = labels