class Recommendation:
    def __init__(self, trace_id, prefix_len, complete_trace, current_prefix, actual_label, target_label, is_compliant,
                 confusion_matrix, impurity, num_samples, fitness, score, recommendation):
        self.trace_id = trace_id
        self.prefix_len = prefix_len
        self.complete_trace = complete_trace
        self.current_prefix = current_prefix
        self.actual_label = actual_label
        self.target_label = target_label
        self.is_compliant = is_compliant
        self.confusion_matrix = confusion_matrix
        self.impurity = impurity
        self.num_samples = num_samples
        self.fitness = fitness
        self.score = score
        self.recommendation = recommendation
