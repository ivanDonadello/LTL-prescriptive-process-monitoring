from src.enums.ConstraintChecker import ConstraintChecker
from src.enums import TraceState
from src.models import TraceResult


# mp-existence constraint checker
# Description:
# The future constraining constraint existence(n, a) indicates that
# event a must occur at least n-times in the trace.
def mp_existence(trace, done, a, b, rules):
    activation_rules = rules["activation"]
    n = rules["n"][ConstraintChecker.EXISTENCE]

    num_activations_in_trace = 0
    for A in trace:
        if A["concept:name"] == a and eval(activation_rules):
            num_activations_in_trace += 1

    state = None
    if not done and num_activations_in_trace < n:
        state = TraceState.POSSIBLY_VIOLATED
    elif done and num_activations_in_trace < n:
        state = TraceState.VIOLATED
    elif num_activations_in_trace >= n:
        state = TraceState.SATISFIED

    traceResult = TraceResult(
        num_fulfillments_in_trace=None,
        num_violations_in_trace=None,
        num_pendings_in_trace=None,
        num_activations_in_trace=None,
        state=state
    )
    return traceResult


# mp-absence constraint checker
# Description:
# The future constraining constraint absence(n + 1, a) indicates that
# event a may occur at most n − times in the trace.
def mp_absence(trace, done, a, b, rules):
    activation_rules = rules["activation"]
    n = rules["n"][ConstraintChecker.ABSENCE]

    num_activations_in_trace = 0
    for A in trace:
        if A["concept:name"] == a and eval(activation_rules):
            num_activations_in_trace += 1

    state = None
    if not done and num_activations_in_trace < n:
        state = TraceState.POSSIBLY_SATISFIED
    elif num_activations_in_trace >= n:
        state = TraceState.VIOLATED
    elif done and num_activations_in_trace < n:
        state = TraceState.SATISFIED

    traceResult = TraceResult(
        num_fulfillments_in_trace=None,
        num_violations_in_trace=None,
        num_pendings_in_trace=None,
        num_activations_in_trace=None,
        state=state
    )
    return traceResult


# mp-init constraint checker
# Description:
# The future constraining constraint init(e) indicates that
# event e is the first event that occurs in the trace.
def mp_init(trace, done, a, b, rules):
    activation_rules = rules["activation"]
    
    state = TraceState.VIOLATED
    if trace[0]["concept:name"] == a:
        A = trace[0]
        if eval(activation_rules):
            state = TraceState.SATISFIED
    traceResult = TraceResult(
        num_fulfillments_in_trace=None,
        num_violations_in_trace=None,
        num_pendings_in_trace=None,
        num_activations_in_trace=None,
        state=state
    )
    return traceResult


# mp-exactly constraint checker
# Description:
def mp_exactly(trace, done, a, b, rules):
    activation_rules = rules["activation"]
    n = rules["n"][ConstraintChecker.EXACTLY]

    num_activations_in_trace = 0
    for A in trace:
        if A["concept:name"] == a and eval(activation_rules):
            num_activations_in_trace += 1

    state = None
    if not done and num_activations_in_trace < n:
        state = TraceState.POSSIBLY_VIOLATED
    elif not done and num_activations_in_trace == n:
        state = TraceState.POSSIBLY_SATISFIED
    elif num_activations_in_trace > n or (done and num_activations_in_trace < n):
        state = TraceState.VIOLATED
    elif done and num_activations_in_trace == n:
        state = TraceState.SATISFIED

    traceResult = TraceResult(
        num_fulfillments_in_trace=None,
        num_violations_in_trace=None,
        num_pendings_in_trace=None,
        num_activations_in_trace=None,
        state=state
    )
    return traceResult
