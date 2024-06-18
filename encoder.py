from argparse import ArgumentParser as argp
import numpy as np

def coorStates(state):
    p1, p2, d, b = int(state[:2]), int(state[2:4]), int(state[4:6]), state[6]
    p1, p2, d = num2xy(p1), num2xy(p2), num2xy(d)
    return addZero(p1) + addZero(p2) + addZero(d) + b

def checkSlope(x1, y1, x2, y2, x3, y3):
    rer = False
    if min(x1,x2) <= x3 <= max(x1,x2) and min(y1,y2) <= y3 <= max(y1,y2):
        if (x1 == x2 == x3) or (y1 == y2 == y3):
            rer = True
        elif ((x1 == x3) and (y1 == y3)) or ((x2 == x3) and (y2 == y3)):
            m = abs((x2-x1)/(y2-y1))
            rer = (m == 1.0)
        else:
            c1 = abs((x1-x3)/(y1-y3)) if (y1-y3) else 'ND'
            c2 = abs((x2-x3)/(y2-y3)) if (y2-y3) else 'ND'
            if c1 == c2 and c1 in eligi:
                rer = True
    return rer

def newStates(state, num, prev_state=None, init_prob=1, defen=False): 
    if defen:
        return state[:4] + addZero(int(state[4:6]) + info[num]) + state[6]

    ball = int(state[-1])
    if 0<=num<=7:
        mov = (num//4) + 1
        int_tmp = addZero(int(state[2*(mov-1):2*mov])+ info[num%4])
        data_xy = state[:2*(mov-1)] + int_tmp + state[2*mov:]
        ep = args.p
        if (0<=num<=3 and ball==1) or (4<=num<=7 and ball==2):
            ep += args.p
            int_prev = state[2*(ball-1):2*ball]
            if int_tmp == data_xy[4:6] or (int_prev == data_xy[4:6] and int_tmp == prev_state[4:6]):
                ep += (1-ep)/2
    elif num==8:
        data_xy = state[:6] + str(int(not (ball-1)) + 1)
        x1, y1, x2, y2, x3, y3 = map(int, data_xy[:-1])
        ep = 1 + 0.1*max(abs(x1-x2), abs(y1-y2)) - args.q
        if checkSlope(x1, y1, x2, y2, x3, y3):
            ep += (1-ep)/2
    elif num==9:
        data_xy = '4444443'
        x_p = int(state[2*ball - 1]) - 1
        ep = 1 + 0.2*(3-x_p) - args.q
        if state[4:6] in ('24','34'):
            ep += (1-ep)/2

    if ('0' in data_xy) or ('5' in data_xy): ep = 1
    mp = 1 - ep
    return data_xy, init_prob*mp, init_prob*ep

addZero = lambda num: '0'+str(num) if num<10 else str(num)
num2xy = lambda n: ((n-1)//4 + 1)*10 + ((n-1)%4 + 1)
xy2num = lambda n: (n//10 - 1)*4 + n%10
info = {0: -1, 1: 1, 2: -10, 3: 10}
eligi = (0.0, 1.0, 'ND')

parser = argp()
parser.add_argument("--opponent", type=str)
parser.add_argument("--p", type=float)
parser.add_argument("--q", type=float)
args = parser.parse_args()

numStates = 16*16*16*2 + 1
numActions = 10
endStates = '1616163'
mdptype = 'episodic'
discount = 1.0
trans = ''

decoder_trans = {coorStates(line.split()[0]): itr for itr, line in enumerate(open(args.opponent).readlines()[1:])}
decoder_trans['4444443'] = 8192

lines = open(args.opponent).readlines()[1:]
for line in lines:
    indo = line.split()
    state, def_prob = coorStates(indo[0]), list(map(float, indo[1:]))
    
    for opt in range(4):
        init_prob = def_prob[opt]
        if init_prob:
            new_stateD = newStates(state, opt, defen=True)
            for action in range(10):
                new_stateP, mp, ep = newStates(new_stateD, action, state, init_prob)
                if mp:
                    r = 1 if new_stateP == '4444443' else 0
                    # print(f'{line}\n{state, def_prob}\n{opt, new_stateD}\n{action, new_stateP}\n{trans}')
                    trans += f'transition {decoder_trans[state]} {action} {decoder_trans[new_stateP]} {r} {mp}\n'

print('numStates', numStates)
print('numActions', numActions)
print('end 8192')
print(trans[:-1])
print('mdptype', mdptype)
print('discount', discount)