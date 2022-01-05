from src.enums import TraceLabel
import pdb
import numpy as np


class Bucket:
    def __init__(self, num_traces=None, num_positive_not_compliant_traces=None, num_positive_compliant_traces=None,
                 num_compliant_traces=None):
        self._num_traces = num_traces
        self._num_positive_not_compliant_traces = num_positive_not_compliant_traces
        self._num_positive_compliant_traces = num_positive_compliant_traces
        self._num_compliant_traces = num_compliant_traces
        self._prefix = None

    def __str__(self):
        return f"Traces:{self._num_traces}, PNCT: {self._num_positive_not_compliant_traces}, " \
               f"PCT: {self._num_positive_compliant_traces}, CT: {self._num_compliant_traces}. Prefix: {self._prefix}"


class Bucketer:
    def __init__(self, bucket_list=[]):
        self.prova = []
        self._bucket_list = bucket_list
        self.smooth_factor = 1
        self.num_classes = 2
        self.total_pos_compl_traces = None
        self.total_pos_not_compl_traces = None
        self.total_compl_traces = None
        self.total_traces = None

    def add_trace(self, prefix, trace_label, compliant):
        found_bucket = False
        if len(self._bucket_list) > 0:
            for bucket in self._bucket_list:
                if prefix == bucket._prefix:
                    bucket._num_traces += 1
                    bucket._num_positive_not_compliant_traces += 0 if compliant else 1
                    bucket._num_positive_compliant_traces += 1 if compliant and trace_label == TraceLabel.TRUE else 0
                    bucket._num_compliant_traces += 1 if compliant else 0
                    found_bucket = True
                    break
        if len(self._bucket_list) == 0 or not found_bucket:
            new_bucket = Bucket()
            new_bucket._num_traces = 1
            new_bucket._num_positive_not_compliant_traces = 0 if compliant else 1
            new_bucket._num_positive_compliant_traces = 1 if compliant and trace_label == TraceLabel.TRUE else 0
            new_bucket._num_compliant_traces = 1 if compliant else 0
            new_bucket._prefix = prefix
            self.add_bucket(new_bucket)

    def add_bucket(self, bucket):
        self._bucket_list.append(bucket)

    def prob_positive_compliant(self):
        self.total_pos_compl_traces = sum([bucket._num_positive_compliant_traces for bucket in self._bucket_list])
        self.total_compl_traces = sum([bucket._num_compliant_traces for bucket in self._bucket_list])
        prob = (self.total_pos_compl_traces + self.smooth_factor) /\
               (self.total_compl_traces + self.smooth_factor*self.num_classes)
        return prob

    def prob_positive_not_compliant(self):
        self.total_pos_not_compl_traces = sum([bucket._num_positive_not_compliant_traces for bucket in self._bucket_list])
        self.total_compl_traces = sum([bucket._num_compliant_traces for bucket in self._bucket_list])
        self.total_traces = sum([bucket._num_traces for bucket in self._bucket_list])

        prob = (self.total_pos_not_compl_traces + self.smooth_factor) / \
               (self.total_traces - self.total_compl_traces + self.smooth_factor*self.num_classes)
        return prob

    def gain(self):

        ee=np.array(self.prova)
        comp = np.sum(ee[:, 1])
        non_comp = len(ee) - comp
        # pdb.set_trace()
        pos_comp = len(np.where((ee == (1, 1)).all(axis=1))[0])
        pos_non_comp = len(np.where((ee == (1, 0)).all(axis=1))[0])
        prob1 = (pos_comp + self.smooth_factor) / (comp + self.smooth_factor*self.num_classes)
        prob2 = (pos_non_comp + self.smooth_factor) / (non_comp + self.smooth_factor*self.num_classes)
        gain = prob1/prob2
        # return self.prob_positive_compliant() / self.prob_positive_not_compliant()
        return gain

    def __str__(self):
        return " | ".join([str(bucket) for bucket in self._bucket_list])