class PathModel(object):

    def __init__(self, impurity, num_samples, rules):
        self.impurity = impurity
        self.num_samples = num_samples
        self.rules = rules