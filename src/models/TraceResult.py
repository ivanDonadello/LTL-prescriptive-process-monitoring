class TraceResult(object):

    def __init__(self, num_fulfillments_in_trace, num_violations_in_trace, num_pendings_in_trace,
                 num_activations_in_trace, state):
        self.num_fulfillments_in_trace = num_fulfillments_in_trace
        self.num_violations_in_trace = num_violations_in_trace
        self.num_pendings_in_trace = num_pendings_in_trace
        self.num_activations_in_trace = num_activations_in_trace
        self.state = state
