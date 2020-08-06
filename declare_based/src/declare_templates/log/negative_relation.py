from declare_based.src.enums import TraceState
from declare_based.src.models import LogResult, TraceResult


# mp-not-responded-existence constraint checker
# Description:
def mp_not_responded_existence(log, done, a, b, activation_rules, correlation_rules):
    print("========== mp-responded-existence constraint checker ==========")
    print("inputs: ")
    print("done: ", done)
    print("a: ", a)
    print("b: ", b)
    print("activation rules: ", activation_rules)
    print("correlation rules: ", correlation_rules)
    print("output: ", end="")
    traces = {}
    num_traces_satisfied_in_log = 0
    num_fulfillments_in_log = 0
    num_violations_in_log = 0
    num_pendings_in_log = 0
    num_activations_in_log = 0
    for trace in log:
        pendings = []
        num_fulfillments_in_trace = 0
        num_violations_in_trace = 0
        num_pendings_in_trace = 0
        for event in trace:
            if event["concept:name"] == a:
                A = event
                if eval(activation_rules):
                    pendings.append(event)
        for event in trace:
            if len(pendings) == 0:
                break
            if event["concept:name"] == b:
                T = event
                for A in reversed(pendings):
                    if eval(correlation_rules):
                        pendings.remove(A)
                        num_violations_in_trace += 1
        if done:
            num_fulfillments_in_trace = len(pendings)
        else:
            num_pendings_in_trace = len(pendings)
        num_activations_in_trace = num_fulfillments_in_trace + num_violations_in_trace + num_pendings_in_trace

        state = None
        if not done and num_violations_in_trace == 0:
            state = TraceState.POSSIBLY_SATISFIED
        elif num_violations_in_trace > 0:
            state = TraceState.VIOLATED
        elif done and num_violations_in_trace == 0:
            num_traces_satisfied_in_log += 1
            state = TraceState.SATISFIED

        traceResult = TraceResult(
            num_fulfillments_in_trace=num_fulfillments_in_trace,
            num_violations_in_trace=num_violations_in_trace,
            num_pendings_in_trace=num_pendings_in_trace,
            num_activations_in_trace=num_activations_in_trace,
            state=state.name
        )
        traces[trace.attributes["concept:name"]] = traceResult.__dict__
        num_fulfillments_in_log += num_fulfillments_in_trace
        num_violations_in_log += num_violations_in_trace
        num_pendings_in_log += num_pendings_in_trace
        num_activations_in_log += num_activations_in_trace
    logResult = LogResult(
        traces=traces,
        num_fulfillments_in_log=num_fulfillments_in_log,
        num_violations_in_log=num_violations_in_log,
        num_pendings_in_log=num_pendings_in_log,
        num_activations_in_log=num_activations_in_log,
        num_traces_satisfied_in_log=num_traces_satisfied_in_log
    )
    print(logResult.__dict__)
    return logResult


# mp-not-response constraint checker
# Description:
def mp_not_response(log, done, a, b, activation_rules, correlation_rules):
    print("========== mp-response constraint checker ==========")
    print("inputs: ")
    print("done: ", done)
    print("a: ", a)
    print("b: ", b)
    print("activation rules: ", activation_rules)
    print("correlation rules: ", correlation_rules)
    print("output: ", end="")
    traces = {}
    num_traces_satisfied_in_log = 0
    num_fulfillments_in_log = 0
    num_violations_in_log = 0
    num_pendings_in_log = 0
    num_activations_in_log = 0
    for trace in log:
        pendings = []
        num_fulfillments_in_trace = 0
        num_violations_in_trace = 0
        num_pendings_in_trace = 0
        for event in trace:
            if event["concept:name"] == a:
                A = event
                if eval(activation_rules):
                    pendings.append(event)
            if len(pendings) > 0 and event["concept:name"] == b:
                T = event
                for A in reversed(pendings):
                    if eval(correlation_rules):
                        pendings.remove(A)
                        num_violations_in_trace += 1
        if done:
            num_fulfillments_in_trace = len(pendings)
        else:
            num_pendings_in_trace = len(pendings)
        num_activations_in_trace = num_fulfillments_in_trace + num_violations_in_trace + num_pendings_in_trace

        state = None
        if not done and num_violations_in_trace == 0:
            state = TraceState.POSSIBLY_SATISFIED
        elif num_violations_in_trace > 0:
            state = TraceState.VIOLATED
        elif done and num_violations_in_trace == 0:
            num_traces_satisfied_in_log += 1
            state = TraceState.SATISFIED

        traceResult = TraceResult(
            num_fulfillments_in_trace=num_fulfillments_in_trace,
            num_violations_in_trace=num_violations_in_trace,
            num_pendings_in_trace=num_pendings_in_trace,
            num_activations_in_trace=num_activations_in_trace,
            state=state.name
        )
        traces[trace.attributes["concept:name"]] = traceResult.__dict__
        num_fulfillments_in_log += num_fulfillments_in_trace
        num_violations_in_log += num_violations_in_trace
        num_pendings_in_log += num_pendings_in_trace
        num_activations_in_log += num_activations_in_trace
    logResult = LogResult(
        traces=traces,
        num_fulfillments_in_log=num_fulfillments_in_log,
        num_violations_in_log=num_violations_in_log,
        num_pendings_in_log=num_pendings_in_log,
        num_activations_in_log=num_activations_in_log,
        num_traces_satisfied_in_log=num_traces_satisfied_in_log
    )
    print(logResult.__dict__)
    return logResult


# mp-not-chain-response constraint checker
# Description:
def mp_not_chain_response(log, done, a, b, activation_rules, correlation_rules):
    print("========== mp-not-chain-response constraint checker ==========")
    print("inputs: ")
    print("done: ", done)
    print("a: ", a)
    print("b: ", b)
    print("activation rules: ", activation_rules)
    print("correlation rules: ", correlation_rules)
    print("output: ", end="")
    traces = {}
    num_traces_satisfied_in_log = 0
    num_fulfillments_in_log = 0
    num_violations_in_log = 0
    num_pendings_in_log = 0
    num_activations_in_log = 0
    for trace in log:
        num_activations_in_trace = 0
        num_violations_in_trace = 0
        num_pendings_in_trace = 0
        for index, event in enumerate(trace):
            if event["concept:name"] == a:
                A = event
                if eval(activation_rules):
                    num_activations_in_trace += 1
                    if index < len(trace) - 1:
                        if trace[index + 1]["concept:name"] == b:
                            T = trace[index + 1]
                            if eval(correlation_rules):
                                num_violations_in_trace += 1
                    else:
                        if not done:
                            num_pendings_in_trace = 1
        num_fulfillments_in_trace = num_activations_in_trace - num_violations_in_trace - num_pendings_in_trace

        state = None
        if not done and num_violations_in_trace == 0:
            state = TraceState.POSSIBLY_SATISFIED
        elif num_violations_in_trace > 0:
            state = TraceState.VIOLATED
        elif done and num_violations_in_trace == 0:
            num_traces_satisfied_in_log += 1
            state = TraceState.SATISFIED

        traceResult = TraceResult(
            num_fulfillments_in_trace=num_fulfillments_in_trace,
            num_violations_in_trace=num_violations_in_trace,
            num_pendings_in_trace=num_pendings_in_trace,
            num_activations_in_trace=num_activations_in_trace,
            state=state.name
        )
        traces[trace.attributes["concept:name"]] = traceResult.__dict__
        num_fulfillments_in_log += num_fulfillments_in_trace
        num_violations_in_log += num_violations_in_trace
        num_pendings_in_log += num_pendings_in_trace
        num_activations_in_log += num_activations_in_trace
    logResult = LogResult(
        traces=traces,
        num_fulfillments_in_log=num_fulfillments_in_log,
        num_violations_in_log=num_violations_in_log,
        num_pendings_in_log=num_pendings_in_log,
        num_activations_in_log=num_activations_in_log,
        num_traces_satisfied_in_log=num_traces_satisfied_in_log
    )
    print(logResult.__dict__)
    return logResult


# mp-not-precedence constraint checker
# Description:
def mp_not_precedence(log, done, a, b, activation_rules, correlation_rules):
    print("========== mp-not-precedence constraint checker ==========")
    print("inputs: ")
    print("done: ", done)
    print("a: ", a)
    print("b: ", b)
    print("activation rules: ", activation_rules)
    print("correlation rules: ", correlation_rules)
    print("output: ", end="")
    traces = {}
    num_traces_satisfied_in_log = 0
    num_fulfillments_in_log = 0
    num_violations_in_log = 0
    num_activations_in_log = 0
    for trace in log:
        num_activations_in_trace = 0
        num_violations_in_trace = 0
        Ts = []
        for event in trace:
            if event["concept:name"] == a:
                Ts.append(event)
            if event["concept:name"] == b:
                A = event
                if eval(activation_rules):
                    num_activations_in_trace += 1
                    for T in Ts:
                        if eval(correlation_rules):
                            num_violations_in_trace += 1
                            break
        num_fulfillments_in_trace = num_activations_in_trace - num_violations_in_trace

        state = None
        if not done and num_violations_in_trace == 0:
            state = TraceState.POSSIBLY_SATISFIED
        elif num_violations_in_trace > 0:
            state = TraceState.VIOLATED
        elif done and num_violations_in_trace == 0:
            num_traces_satisfied_in_log += 1
            state = TraceState.SATISFIED

        traceResult = TraceResult(
            num_fulfillments_in_trace=num_fulfillments_in_trace,
            num_violations_in_trace=num_violations_in_trace,
            num_pendings_in_trace=None,
            num_activations_in_trace=num_activations_in_trace,
            state=state.name
        )
        traces[trace.attributes["concept:name"]] = traceResult.__dict__
        num_fulfillments_in_log += num_fulfillments_in_trace
        num_violations_in_log += num_violations_in_trace
        num_activations_in_log += num_activations_in_trace
    logResult = LogResult(
        traces=traces,
        num_fulfillments_in_log=num_fulfillments_in_log,
        num_violations_in_log=num_violations_in_log,
        num_pendings_in_log=None,
        num_activations_in_log=num_activations_in_log,
        num_traces_satisfied_in_log=num_traces_satisfied_in_log
    )
    print(logResult.__dict__)
    return logResult


# mp-not-chain-precedence constraint checker
# Description:
def mp_not_chain_precedence(log, done, a, b, activation_rules, correlation_rules):
    print("========== mp-not-chain-precedence constraint checker ==========")
    print("inputs: ")
    print("done: ", done)
    print("a: ", a)
    print("b: ", b)
    print("activation rules: ", activation_rules)
    print("correlation rules: ", correlation_rules)
    print("output: ", end="")
    traces = {}
    num_traces_satisfied_in_log = 0
    num_fulfillments_in_log = 0
    num_violations_in_log = 0
    num_activations_in_log = 0
    for trace in log:
        num_activations_in_trace = 0
        num_violations_in_trace = 0
        for index, event in enumerate(trace):
            if event["concept:name"] == b:
                A = event
                if eval(activation_rules):
                    num_activations_in_trace += 1
                    if index != 0 and trace[index - 1]["concept:name"] == a:
                        T = trace[index - 1]
                        if eval(correlation_rules):
                            num_violations_in_trace += 1
        num_fulfillments_in_trace = num_activations_in_trace - num_violations_in_trace

        state = None
        if not done and num_violations_in_trace == 0:
            state = TraceState.POSSIBLY_SATISFIED
        elif num_violations_in_trace > 0:
            state = TraceState.VIOLATED
        elif done and num_violations_in_trace == 0:
            num_traces_satisfied_in_log += 1
            state = TraceState.SATISFIED

        traceResult = TraceResult(
            num_fulfillments_in_trace=num_fulfillments_in_trace,
            num_violations_in_trace=num_violations_in_trace,
            num_pendings_in_trace=None,
            num_activations_in_trace=num_activations_in_trace,
            state=state.name
        )
        traces[trace.attributes["concept:name"]] = traceResult.__dict__
        num_fulfillments_in_log += num_fulfillments_in_trace
        num_violations_in_log += num_violations_in_trace
        num_activations_in_log += num_activations_in_trace
    logResult = LogResult(
        traces=traces,
        num_fulfillments_in_log=num_fulfillments_in_log,
        num_violations_in_log=num_violations_in_log,
        num_pendings_in_log=None,
        num_activations_in_log=num_activations_in_log,
        num_traces_satisfied_in_log=num_traces_satisfied_in_log
    )
    print(logResult.__dict__)
    return logResult
