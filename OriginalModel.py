import numpy as np
from gurobipy import *
from itertools import product
import math


bigM = 1000
def originalModel(S, N, O, H, P, B, ES, OT, RT, L, DT, CT, K, lb, timeLimit):
    m = Model()
    x = m.addVars(N, N, lb=0, vtype=GRB.BINARY, name='x')
    y = m.addVars(S, N, lb=0, vtype=GRB.INTEGER, name='y')
    z = m.addVars(S, N, lb=0, vtype=GRB.INTEGER, name='z')
    pt = m.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name='pt')
    ot = m.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name='ot')
    r = m.addVars(N, N, lb=0, vtype=GRB.BINARY, name='r')
    ct = m.addVars(N, lb=0, vtype=GRB.CONTINUOUS, name='ot')
    c_max = m.addVar(lb=lb, vtype=GRB.CONTINUOUS)

    m.setObjective(c_max, sense=GRB.MINIMIZE)

    #constraint 1: 每个阶段移出一个钢板
    m.addConstrs(x.sum('*', n) == 1 for n in range(N))
    #constraint 2: 每张板只在一个阶段移出
    m.addConstrs(x.sum(i, '*') == 1 for i in range(N))
    #constraint 3: 位于堆位上方的钢板要先移出
    for i, n in product(range(N), range(N)):
        if i in O.keys(): m.addConstrs( x[i, n] <= quicksum(x[j, n1] for n1 in range(n + 1)) for j in O[i])
    #constraint 5: 当前阶段没有出库钢板的堆位 y=0
    m.addConstrs(y[s,n] <= H * quicksum(x[i, n] * P[i, s] for i in range(N)) for s in range(S) for n in range(N) )
    #constraint 6，7: 每个阶段每个堆位y的取值
    for n, s in product(range(N), range(S)):
        linExpr1 = quicksum(x[i, n] * P[i, s] for i in range(N)) # x*p
        linExpr2 = quicksum(x[i, n] * P[i, s] * B[i] for i in range(N)) # x*p*b
        linExpr3 = LinExpr(0)
        if n > 0: linExpr3 = quicksum(y[s, n1] - z[s, n1] for n1 in range(n))
        m.addConstr(y[s, n] + linExpr3 <= linExpr2 + bigM - bigM * linExpr1)
        m.addConstr(y[s, n] + linExpr3 >= linExpr2 - bigM + bigM * linExpr1)
        del linExpr3, linExpr2, linExpr1
    #constraint 9: 有出库的堆位不能接收钢板
    m.addConstrs( z[s,n] <= H * (1- quicksum(x[i,n] * P[i, s] for i in range(N))) for s in range(S) for n in range(N) )
    #constratin 10: 每个堆位接收BP上限
    for n, s in product(range(N), range(S)):
        linExpr4 = LinExpr(0)
        if n > 0:
            linExpr4 = quicksum(y[s, n1] - z[s, n1] for n1 in range(n))
            linExpr4.add(quicksum(x[i, n1] * P[i, s] for i in range(n) for n1 in range(n)))
        m.addConstr(z[s,n] <= ES[s] + linExpr4)
        del linExpr4
    #constraint 11: y=z
    m.addConstrs( z.sum('*', n) == y.sum('*', n) for n in range(N) )
    #constraint 12，13: 每个阶段的移动时间
    m.addConstrs(pt[n] <= OT[i] + bigM * (1 - x[i, n] * P[i, s]) + quicksum(z[s1, n] * RT[s, s1] for s1 in range(S)) \
                 for i in range(N) for n in range(N) for s in range(S))
    m.addConstrs(pt[n] >= OT[i] - bigM * (1 - x[i, n] * P[i, s]) + quicksum(z[s1, n] * RT[s, s1] for s1 in range(S)) \
                 for i in range(N) for n in range(N) for s in range(S))
    #constraint 14，15： 每张板出库时间
    m.addConstrs(ot[i] <= quicksum(pt[n1] for n1 in range(n + 1)) + bigM * (1 - x[i, n]) \
                 for i in range(N) for n in range(N))
    m.addConstrs(ot[i] >= quicksum(pt[n1] for n1 in range(n + 1)) - bigM * (1 - x[i, n]) \
                 for i in range(N) for n in range(N))
    #constraint 16: 钢板切割顺序和出库时间的关系
    m.addConstrs( ot[i] <= ot[j] + bigM * (1-r[i,j]) for i in range(N) for j in range(N) )
    #constraint 17，18：同一条切割线才有先后关系,自身没有先后关系
    for i in range(N):
        m.addConstr(r[i, i] == 0)
        for j in range(i+1,N):
            m.addConstr( r[i,j] + r[j,i] == K[i,j] )

    #constraint 19，20：切割完成时间，大于到达时间，大于前一个任务完成时间
    m.addConstrs(ct[i] >= ot[i] + DT + CT[i] for i in range(N))
    m.addConstrs(ct[i] >= ct[j] + CT[i] - bigM * (1 - r[j, i]) for i in range(N) for j in range(N))

    #constraint 21：makespan计算
    m.addConstrs(c_max >= ct[i] for i in range(N))
    #m.update()
    m.Params.OutputFlag = 1
    m.Params.TimeLimit = timeLimit
    m.Params.Threads = 8
    #m2.setParam('OutputFlag', False)
    m.optimize()
    return m.objval, m.Status, m.MIPGap, m.Runtime