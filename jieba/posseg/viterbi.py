import sys
import operator

MIN_FLOAT = -3.14e100
MIN_INF = float("-inf")

if sys.version_info[0] > 2:
    xrange = range


def get_top_states(t_state_v, K=4):
    return sorted(t_state_v, key=t_state_v.__getitem__, reverse=True)[:K]


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # tabular
    mem_path = [{}]
    # 根据状态转移矩阵，获取所有的可能性
    all_states = trans_p.keys()
    for y in states.get(obs[0], all_states):  # init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        mem_path[0][y] = ''
    # 时刻t=1,...len(obs)
    for t in xrange(1, len(obs)):
        V.append({})
        mem_path.append({})
        # prev_states = get_top_states(V[t-1])
        # 获取前一刻所有状态集合
        prev_states = [
            x for x in mem_path[t - 1].keys() if len(trans_p[x]) > 0]
        # 根据前一刻的状态和状态转移矩阵，提前计算当前时刻的状态集合
        prev_states_expect_next = set(
            (y for x in prev_states for y in trans_p[x].keys()))
        # 根据当前的观察值获得当前时刻的可能状态集合，在与上一步计算的状态集合取交集
        obs_states = set(
            states.get(obs[t], all_states)) & prev_states_expect_next
        # 当前时刻交集的集合为空
        if not obs_states:
            # 如果提前计算当前时刻计算状态不为空，则当前时刻的状态集合为提前计算时刻的状态集合，否则为全部的可能状态集合
            obs_states = prev_states_expect_next if prev_states_expect_next else all_states
        # 当前时刻所处的各种可能状态集合
        for y in obs_states:
            # 分别获取上一时刻的状态的概率对数，该状态到本时刻的状态的转移概率对数，本时刻的状态的发射概率对数
            # prev_states是当前时刻的状态所对应上一时刻可能的状态集合
            prob, state = max((V[t - 1][y0] + trans_p[y0].get(y, MIN_INF) +
                               emit_p[y].get(obs[t], MIN_FLOAT), y0) for y0 in
                              prev_states)
            V[t][y] = prob
            mem_path[t][y] = state
    # 最后一个时刻
    last = [(V[-1][y], y) for y in mem_path[-1].keys()]
    # if len(last)==0:
    #     print obs
    prob, state = max(last)

    route = [None] * len(obs)
    i = len(obs) - 1
    while i >= 0:
        route[i] = state
        state = mem_path[i][state]
        i -= 1
    # 返回最大概率以及各个时刻的状态
    return (prob, route)
