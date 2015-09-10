""" Splits a Seq2Seq-format dataset into train,dev[,test] splits """
import sys
import os

if len(sys.argv) < 4:
    raise "Usage: python split.py <data_dir> <train_split> <dev_split> [test_split]"

train_split = float(sys.argv[2])
dev_split = float(sys.argv[3])
test_split = 0.
if len(sys.argv) ==5:
    test_split = float(sys.argv[4])

assert train_split+dev_split+test_split==1., "train,dev,test split must add to 1; instead, got %s." % (str(train_split+dev_split+test_split))

sf = open(os.path.join(sys.argv[1], 'source.txt'))
tf = open(os.path.join(sys.argv[1], 'target.txt'))

slines = []
tlines = []
def full_strip(x):
    return x.rstrip('\n').rstrip('\r').rstrip('\n')

for line in sf:
    slines.append(full_strip(line).lower())

for line in tf:
    tlines.append(full_strip(line).lower())

assert len(slines) == len(tlines), "source and target lengths didn't match. source: %s lines. target: %s lines." % (str(len(slines)), str(len(tlines)))

train_split_idx = int(len(tlines)*train_split)
if test_split>0:
    dev_split_idx = int(len(tlines)*(train_split+dev_split))
    train_slines = slines[:train_split_idx]
    train_tlines = tlines[:train_split_idx]
    dev_slines = slines[train_split_idx:dev_split_idx]
    dev_tlines = tlines[train_split_idx:dev_split_idx]
    test_slines = slines[dev_split_idx:]
    test_tlines = tlines[dev_split_idx:]

    os.system('mkdir %s' % os.path.join(sys.argv[1], 'train'))
    os.system('mkdir %s' % os.path.join(sys.argv[1], 'dev'))
    os.system('mkdir %s' % os.path.join(sys.argv[1], 'test'))

    with open(os.path.join(sys.argv[1], 'train/source.txt'),'w') as f:
        f.write('\n'.join(train_slines))
    with open(os.path.join(sys.argv[1], 'train/target.txt'),'w') as f:
        f.write('\n'.join(train_tlines))
    with open(os.path.join(sys.argv[1], 'dev/source.txt'),'w') as f:
        f.write('\n'.join(dev_slines))
    with open(os.path.join(sys.argv[1], 'dev/target.txt'),'w') as f:
        f.write('\n'.join(dev_tlines))
    with open(os.path.join(sys.argv[1], 'test/source.txt'),'w') as f:
        f.write('\n'.join(test_slines))
    with open(os.path.join(sys.argv[1], 'test/target.txt'),'w') as f:
        f.write('\n'.join(test_tlines))



else:
    train_slines = slines[:train_split_idx]
    train_tlines = tlines[:train_split_idx]
    dev_slines = slines[train_split_idx:]
    dev_tlines = tlines[train_split_idx:]

    os.system('mkdir %s' % os.path.join(sys.argv[1], 'train'))
    os.system('mkdir %s' % os.path.join(sys.argv[1], 'dev'))

    with open(os.path.join(sys.argv[1], 'train/source.txt'),'w') as f:
        f.write('\n'.join(train_slines))
    with open(os.path.join(sys.argv[1], 'train/target.txt'),'w') as f:
        f.write('\n'.join(train_tlines))
    with open(os.path.join(sys.argv[1], 'dev/source.txt'),'w') as f:
        f.write('\n'.join(dev_slines))
    with open(os.path.join(sys.argv[1], 'dev/target.txt'),'w') as f:
        f.write('\n'.join(dev_tlines))


print "Success."

    

