'''
翻板子问题。
在给定出库顺序后，首先计算一些中间变量。
之后建立翻板数学模型，得到最终的结果以及每个钢板的出库以及切割完成时间。
'''
import time

from gurobipy import *
import numpy as np
from itertools import product

def interValue(pi, CutLineOfPlate, StackOfPlate, L, N):
    #每条线最后一个出库任务
    lastPlateOfLine = [-1 for i in range(L)]
    #每张板出库时段
    outboundPeriodOfPlate = dict()
    #xBar
    xBar = np.zeros((N, N), dtype=int)
    #紧前切割任务，没键值说明是切割线的第一个任务
    cutFormerPlate = dict()
    #紧前出库任务，没键值说明是第一个出库的任务
    outboundFormerPlate = dict()
    for index, plate in enumerate(pi):
        #当前钢板的切割线和堆位
        plateLine = CutLineOfPlate[plate]
        plateStack = StackOfPlate[plate]
        # 切割先后顺序
        if lastPlateOfLine[plateLine] != -1: cutFormerPlate[plate] = lastPlateOfLine[plateLine]
        lastPlateOfLine[plateLine] = plate

        #每张钢板的出库时间
        outboundPeriodOfPlate[plate] = index
        xBar[plate][index] = 1
        #紧前出库任务
        if index != 0: outboundFormerPlate[plate] = pi[index - 1]

    return outboundPeriodOfPlate, xBar, outboundFormerPlate, cutFormerPlate, lastPlateOfLine
def RSP(pi, CutLineOfPlate, StackOfPlate, L, N, S, DT, CT, OT, ES, P, B, RT):
    #中间变量
    #print('in RSP',time.time())
    outboundPeriodOfPlate, xBar, outboundFormerPlate, cutFormerPlate, lastPlateOfLine = \
        interValue(pi, CutLineOfPlate, StackOfPlate, L, N)

    #建模
    m = Model()
    #变量
    y = m.addVars(S, N, lb=0, vtype=GRB.INTEGER, name='y')
    z = m.addVars(S, N, lb=0, vtype=GRB.INTEGER, name='z')
    ot = m.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name='ot')
    ct = m.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name='ct')
    c_max = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='c_max')
    #目标
    m.setObjectiveN(c_max, 0, 2)
    m.setObjectiveN(quicksum(ct) + quicksum(ot), 1, 1)
    #约束
    #constraint rsp 1-5: 移入移出量
    for n, s in product(range(N), range(S)):
        if s != StackOfPlate[pi[n]]:
            # constraint rsp-1: 不是出库堆位的不能移出钢板
            m.addConstr(y[s, n] == 0)
            # constraint rsp-5: 不是出库堆位的移入钢板限制
            if n > 0:
                linExpr1 = quicksum(z[s, n1] for n1 in range(n))
                linExpr2 = quicksum(y[s, n1] for n1 in range(n))
                additional = sum(P[i, s] * xBar[i, n1] for i in range(N) for n1 in range(n))
                m.addConstr(z[s, n] <= ES[s] + linExpr2 - linExpr1 + additional)
            else:
                m.addConstr(z[s, n] <= ES[s])
        else:
            # constraint rsp-3: 出库堆位的钢板移入数量等于0
            m.addConstr(z[s, n] == 0)
            # constraint rsp-2: 出库堆位的钢板移出数量
            if n > 0:
                linExpr1 = quicksum(z[s, n1] for n1 in range(n))
                linExpr2 = quicksum(y[s, n1] for n1 in range(n))
                m.addConstr(y[s, n] == B[pi[n]] + linExpr1 - linExpr2)
            else:
                m.addConstr(y[s, n] == B[pi[n]])
    # constraint rsp-6: 每个钢板的ot值
    for i in range(N):
        index = pi.index(i)  # 出库顺序
        outboundStack = StackOfPlate[i] #出库堆位
        if index == 0:
            m.addConstr(ot[i] == OT[i] + quicksum(RT[outboundStack, s] * z[s, 0] for s in range(S)))
        else:
            m.addConstr(ot[i] == ot[pi[index - 1]] + OT[i] + quicksum(
                RT[outboundStack, s] * z[s, index] for s in range(S)))
    # constraint rsp-7: ct[i]>=ct[j]+CT[i]
    for i in range(N):
        if i in cutFormerPlate.keys():
            m.addConstr(ct[i] >= ct[cutFormerPlate[i]] + CT[i])
    # constraint rsp-8: c_max >=每条线最后一个任务的完成时间
    for i in lastPlateOfLine:
        if i != -1:
            m.addConstr(c_max >= ct[i])
    # constraint 22: y=z
    m.addConstrs(z.sum('*', n) == y.sum('*', n) for n in range(N))
    # constraint 30: ct[i]>=ot[i]+DT+CT[i]
    m.addConstrs(ct[i] >= ot[i] + DT + CT[i] for i in range(N))
    #求解
    m.setParam('OutputFlag', 0)
    m.optimize()
    #输出结果
    outboundTime = dict()
    cutTime = dict()
    for n in range(N):
        #print('\nstage', n, 'stack is ', StackOfPlate[pi[n]] + 1)
        # for s in range(S):
        #     #if round(y[s, n].x) >= 1: print(round(y[s, n].x), 'plates are relocated from stack', s )
        #     if round(z[s, n].x) >= 1: print(round(z[s, n].x), 'plates are relocated into stack', s + 1 )
        # print('outboundPlate is', pi[n], ',it\'s outbound time is', ot[pi[n]].x, ', it\'s cut time is ',
        #       ct[pi[n]].x)
        outboundTime[pi[n]] = ot[pi[n]].x
        cutTime[pi[n]] = ct[pi[n]].x
    return round(m.objVal, 6), outboundTime, cutTime
#只输出结果，没有其他乱七八糟的
def simplifiedRSP(pi, CutLineOfPlate, StackOfPlate, L, N, S, DT, CT, OT, ES, P, B, RT):
    #中间变量
    outboundPeriodOfPlate, xBar, outboundFormerPlate, cutFormerPlate, lastPlateOfLine = \
        interValue(pi, CutLineOfPlate, StackOfPlate, L, N)

    #建模
    m = Model()
    #变量
    y = m.addVars(S, N, lb=0, vtype=GRB.INTEGER, name='y')
    z = m.addVars(S, N, lb=0, vtype=GRB.INTEGER, name='z')
    ot = m.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name='ot')
    ct = m.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name='ct')
    c_max = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='c_max')
    #目标
    m.setObjectiveN(c_max, 0, 2)
    m.setObjectiveN(quicksum(ct) + quicksum(ot), 1, 1)
    #约束
    #constraint rsp 1-5: 移入移出量
    for n, s in product(range(N), range(S)):
        if s != StackOfPlate[pi[n]]:
            # constraint rsp-1: 不是出库堆位的不能移出钢板
            m.addConstr(y[s, n] == 0)
            # constraint rsp-5: 不是出库堆位的移入钢板限制
            if n > 0:
                linExpr1 = quicksum(z[s, n1] for n1 in range(n))
                linExpr2 = quicksum(y[s, n1] for n1 in range(n))
                additional = sum(P[i, s] * xBar[i, n1] for i in range(N) for n1 in range(n))
                m.addConstr(z[s, n] <= ES[s] + linExpr2 - linExpr1 + additional)
            else:
                m.addConstr(z[s, n] <= ES[s])
        else:
            # constraint rsp-3: 出库堆位的钢板移入数量等于0
            m.addConstr(z[s, n] == 0)
            # constraint rsp-2: 出库堆位的钢板移出数量
            if n > 0:
                linExpr1 = quicksum(z[s, n1] for n1 in range(n))
                linExpr2 = quicksum(y[s, n1] for n1 in range(n))
                m.addConstr(y[s, n] == B[pi[n]] + linExpr1 - linExpr2)
            else:
                m.addConstr(y[s, n] == B[pi[n]])
    # constraint rsp-6: 每个钢板的ot值
    for i in range(N):
        index = pi.index(i)  # 出库顺序
        outboundStack = StackOfPlate[i] #出库堆位
        if index == 0:
            m.addConstr(ot[i] == OT[i] + quicksum(RT[outboundStack, s] * z[s, 0] for s in range(S)))
        else:
            m.addConstr(ot[i] == ot[pi[index - 1]] + OT[i] + quicksum(
                RT[outboundStack, s] * z[s, index] for s in range(S)))
    # constraint rsp-7: ct[i]>=ct[j]+CT[i]
    for i in range(N):
        if i in cutFormerPlate.keys():
            m.addConstr(ct[i] >= ct[cutFormerPlate[i]] + CT[i])
    # constraint rsp-8: c_max >=每条线最后一个任务的完成时间
    for i in lastPlateOfLine:
        if i != -1:
            m.addConstr(c_max >= ct[i])
    # constraint 22: y=z
    m.addConstrs(z.sum('*', n) == y.sum('*', n) for n in range(N))
    # constraint 30: ct[i]>=ot[i]+DT+CT[i]
    m.addConstrs(ct[i] >= ot[i] + DT + CT[i] for i in range(N))
    #求解
    m.setParam('OutputFlag', 0)
    m.optimize()
    return round(m.objVal, 6)
#输出全部内容的rsp
def RSP1(pi, CutLineOfPlate, StackOfPlate, L, N, S, DT, CT, OT, ES, P, B, RT):
    #中间变量
    outboundPeriodOfPlate, xBar, outboundFormerPlate, cutFormerPlate, lastPlateOfLine = \
        interValue(pi, CutLineOfPlate, StackOfPlate, L, N)

    #建模
    m = Model()
    #变量
    y = m.addVars(S, N, lb=0, vtype=GRB.INTEGER, name='y')
    z = m.addVars(S, N, lb=0, vtype=GRB.INTEGER, name='z')
    ot = m.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name='ot')
    ct = m.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name='ct')
    c_max = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='c_max')
    #目标
    m.setObjectiveN(c_max, 0, 2)
    m.setObjectiveN(quicksum(ct) + quicksum(ot), 1, 1)
    #约束
    #constraint rsp 1-5: 移入移出量
    for n, s in product(range(N), range(S)):
        if s != StackOfPlate[pi[n]]:
            # constraint rsp-1: 不是出库堆位的不能移出钢板
            m.addConstr(y[s, n] == 0)
            # constraint rsp-5: 不是出库堆位的移入钢板限制
            if n > 0:
                linExpr1 = quicksum(z[s, n1] for n1 in range(n))
                linExpr2 = quicksum(y[s, n1] for n1 in range(n))
                additional = sum(P[i, s] * xBar[i, n1] for i in range(N) for n1 in range(n))
                m.addConstr(z[s, n] <= ES[s] + linExpr2 - linExpr1 + additional)
            else:
                m.addConstr(z[s, n] <= ES[s])
        else:
            # constraint rsp-3: 出库堆位的钢板移入数量等于0
            m.addConstr(z[s, n] == 0)
            # constraint rsp-2: 出库堆位的钢板移出数量
            if n > 0:
                linExpr1 = quicksum(z[s, n1] for n1 in range(n))
                linExpr2 = quicksum(y[s, n1] for n1 in range(n))
                m.addConstr(y[s, n] == B[pi[n]] + linExpr1 - linExpr2)
            else:
                m.addConstr(y[s, n] == B[pi[n]])
    # constraint rsp-6: 每个钢板的ot值
    for i in range(N):
        index = pi.index(i)  # 出库顺序
        outboundStack = StackOfPlate[i] #出库堆位
        if index == 0:
            m.addConstr(ot[i] == OT[i] + quicksum(RT[outboundStack, s] * z[s, 0] for s in range(S)))
        else:
            m.addConstr(ot[i] == ot[pi[index - 1]] + OT[i] + quicksum(
                RT[outboundStack, s] * z[s, index] for s in range(S)))
    # constraint rsp-7: ct[i]>=ct[j]+CT[i]
    for i in range(N):
        if i in cutFormerPlate.keys():
            m.addConstr(ct[i] >= ct[cutFormerPlate[i]] + CT[i])
    # constraint rsp-8: c_max >=每条线最后一个任务的完成时间
    m.addConstrs(c_max >= ct[i] for i in lastPlateOfLine)
    # constraint 22: y=z
    m.addConstrs(z.sum('*', n) == y.sum('*', n) for n in range(N))
    # constraint 30: ct[i]>=ot[i]+DT+CT[i]
    m.addConstrs(ct[i] >= ot[i] + DT + CT[i] for i in range(N))
    #求解
    m.setParam('OutputFlag', 0)
    m.optimize()
    #输出结果
    outboundTime = dict()
    cutTime = dict()
    yRes = np.zeros((S,N),dtype=int)
    zRes = np.zeros((S,N),dtype=int)
    for n in range(N):
        print('\nstage', n, 'stack is ', StackOfPlate[pi[n]] + 1)
        for s in range(S):
            if round(y[s, n].x) >= 1: yRes[s,n] = round(y[s, n].x)
            if round(z[s, n].x) >= 1: zRes[s,n] = round(z[s, n].x)
        outboundTime[pi[n]] = ot[pi[n]].x
        cutTime[pi[n]] = ct[pi[n]].x
    return round(m.objVal, 6), outboundTime, cutTime, yRes, zRes































