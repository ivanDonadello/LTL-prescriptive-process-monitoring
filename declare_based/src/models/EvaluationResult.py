class EvaluationResult(object):

    def __init__(self, tp=0, fp=0, tn=0, fn=0, precision = 0, recall = 0, accuracy = 0, fscore = 0, auc = 0):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.fscore = fscore
        self.auc = auc