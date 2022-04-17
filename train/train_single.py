import sys
from model import *
from torch import randperm
from torch._utils import _accumulate
#from utils_torch import progress_bar
from train_multiple import crossent, train, test

assert torch.cuda.is_available()

N_GPUS = 1
N_CORES = 6
BATCH_SIZE = 12
model_num = int(sys.argv[1])
TISSUE = int(sys.argv[3])
torch.backends.cudnn.benchmark = True

def random_split(dataset, lengths):
    indices = randperm(sum(lengths)).tolist()
    for offset, length in zip(_accumulate(lengths), lengths):
        return indices[offset - length:offset]

ds = H5Dataset(sys.argv[4])
val = round(0.1*len(ds))
train = len(ds)-val
train_ds, val_ds = data.random_split(ds, (train,val))

train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES, pin_memory=True)
val_dl = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES, pin_memory=True)

model = Pangolin(L, W, AR)
if torch.cuda.device_count() > 1:
    print("Using %s gpus" % torch.cuda.device_count())
    model = nn.DataParallel(model)
model.cuda()
model.load_state_dict(torch.load(sys.argv[2]))

bce = nn.BCELoss()

def loss(y_pred, y_true):
    y_pred = torch.split(y_pred, [2,1,2,1,2,1,2,1], dim=1)
    y_true = torch.split(y_true, [2,1,2,1,2,1,2,1], dim=1)

    if TISSUE % 2 == 0:
        loss = crossent(y_pred[TISSUE], y_true[TISSUE])
    else:
        loss = bce(y_pred[TISSUE][y_true[TISSUE]>=0], y_true[TISSUE][y_true[TISSUE]>=0])

    return loss

criterion = loss
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
T_0 = 4
T_mult = 2
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
iters = len(train_dl)
flog_final = open("log.%s.%s.txt" % (model_num, TISSUE), 'w')

for epoch in range(0, 4):
    train_loss = train(epoch)
    test_loss = test()
    print(epoch, train_loss, test_loss, file=flog_final)
    flog_final.flush()
    torch.save(model.state_dict(), "models/final.%s.%s.%s" % (model_num, TISSUE, epoch))



