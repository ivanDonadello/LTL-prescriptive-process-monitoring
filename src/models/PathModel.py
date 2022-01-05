class PathModel:
    def __init__(self, impurity, num_samples, rules):
        self.impurity = impurity
        self.num_samples = num_samples
        self.rules = rules
        self.fitness = None
        self.score = None