from declare_based.src.enums import TraceState
from declare_based.src.models import LogResult, TraceResult


# mp-choice constraint checker
# Description:
def mp_choice(log, done, a, b, activation_rules):
    print("========== mp-choice constraint checker ==========")
    print("inputs: ")
    print("done: ", done)
    print("a: ", a)
    print("b: ", b)
    print("activation rules: ", activation_rules)
    print("output: ", end="")
    traces = {}
    num_traces_satisfied_in_log = 0
    for trace in log:
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
            num_traces_satisfied_in_log += 1
            state = TraceState.SATISFIED

        traceResult = TraceResult(
            num_fulfillments_in_trace=None,
            num_violations_in_trace=None,
            num_pendings_in_trace=None,
            num_activations_in_trace=None,
            state=state.name
        )
        traces[trace.attributes["concept:name"]] = traceResult.__dict__
    logResult = LogResult(
        traces=traces,
        num_fulfillments_in_log=None,
        num_violations_in_log=None,
        num_pendings_in_log=None,
        num_activations_in_log=None,
        num_traces_satisfied_in_log=num_traces_satisfied_in_log
    )
    print(logResult.__dict__)
    return logResult


# mp-exclusive-choice constraint checker
# Description:
def mp_exclusive_choice(log, done, a, b, activation_rules):
    print("========== mp-exclusive-choice constraint checker ==========")
    print("inputs: ")
    print("done: ", done)
    print("a: ", a)
    print("b: ", b)
    print("activation rules: ", activation_rules)
    print("output: ", end="")
    traces = {}
    num_traces_satisfied_in_log = 0
    for trace in log:
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
            num_traces_satisfied_in_log += 1
            state = TraceState.SATISFIED

        traceResult = TraceResult(
            num_fulfillments_in_trace=None,
            num_violations_in_trace=None,
            num_pendings_in_trace=None,
            num_activations_in_trace=None,
            state=state.name
        )
        traces[trace.attributes["concept:name"]] = traceResult.__dict__
    logResult = LogResult(
        traces=traces,
        num_fulfillments_in_log=None,
        num_violations_in_log=None,
        num_pendings_in_log=None,
        num_activations_in_log=None,
        num_traces_satisfied_in_log=num_traces_satisfied_in_log
    )
    print(logResult.__dict__)
    return logResult
