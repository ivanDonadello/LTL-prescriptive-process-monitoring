from src.enums import TraceState
from src.models import CheckerResult


# mp-choice constraint checker
# Description:
def mp_choice(trace, done, a, b, rules):
    activation_rules = rules["activation"]

    a_or_b_occurs = False
    for A in trace:
        if A["concept:name"] == a or A["concept:name"] == b:
            if eval(activation_rules):
                a_or_b_occurs = True
                break

    state = None
    if not done and not a_or_b_occurs:
        state = TraceState.POSSIBLY_VIOLATED
    elif done and not a_or_b_occurs:
        state = TraceState.VIOLATED
    elif a_or_b_occurs:
        state = TraceState.SATISFIED

    return CheckerResult(num_fulfillments=None, num_violations=None, num_pendings=None, num_activations=None, state=state)


# mp-exclusive-choice constraint checker
# Description:
def mp_exclusive_choice(trace, done, a, b, rules):
    activation_rules = rules["activation"]

    a_occurs = False
    b_occurs = False
    for A in trace:
        if not a_occurs and A["concept:name"] == a:
            if eval(activation_rules):
                a_occurs = True
        if not b_occurs and A["concept:name"] == b:
            if eval(activation_rules):
                b_occurs = True
        if a_occurs and b_occurs:
            break

    state = None
    if not done and (not a_occurs and not b_occurs):
        state = TraceState.POSSIBLY_VIOLATED
    elif not done and (a_occurs ^ b_occurs):
        state = TraceState.POSSIBLY_SATISFIED
    elif (a_occurs and b_occurs) or (done and (not a_occurs and not b_occurs)):
        state = TraceState.VIOLATED
    elif done and (a_occurs ^ b_occurs):
        state = TraceState.SATISFIED

    return CheckerResult(num_fulfillments=None, num_violations=None, num_pendings=None, num_activations=None, state=state)
