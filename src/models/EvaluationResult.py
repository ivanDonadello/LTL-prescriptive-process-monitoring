class EvaluationResult:
    def __init__(self, tp=0, fp=0, tn=0, fn=0, c=0, nc=0, pc=0, pnc=0, precision=0, recall=0, accuracy=0, fscore=0, auc=0, mcc=0, gain=0):
        self.num_cases = None
        self.prefix_length = None

        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

        self.comp = c
        self.non_comp = nc
        self.pos_comp = pc
        self.pos_non_comp = pnc

        self.precision = precision
        self.recall = recall
        self.accuracy = accuracy
        self.fscore = fscore
        self.auc = auc
        self.mcc = mcc
        self.gain = gain

        self.th = -1
        self.fscore_list = []
