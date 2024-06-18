from argparse import ArgumentParser as argp
parser = argp()
parser.add_argument("--value-policy", type=str)
parser.add_argument("--opponent", type=str)
args = parser.parse_args()

decoder_trans = {itr: line.split()[0] for itr, line in enumerate(open(args.opponent).readlines()[1:])}
lines = open(args.value_policy).readlines()[:-1]
for itr, line in enumerate(lines):
    temp = line.strip().split()
    print(f'{decoder_trans[itr]} {temp[1]} {temp[0]}')