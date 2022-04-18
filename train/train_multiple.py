import sys
from model import *
from torch import randperm
from torch._utils import _accumulate
from utils_torch import progress_bar

assert torch.cuda.is_available()

N_GPUS = 1
N_CORES = 6
BATCH_SIZE = 12
model_num = int(sys.argv[1])
torch.backends.cudnn.benchmark = True

def random_split(dataset, lengths):
    indices = randperm(sum(lengths)).tolist()
    for offset, length in zip(_accumulate(lengths), lengths):
        return indices[offset - length:offset]

ds = H5Dataset(sys.argv[2])
val = round(0.1*len(ds))
train = len(ds)-val
train_ds, val_ds = data.random_split(ds, (train,val))

train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES, pin_memory=True)
val_dl = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES, pin_memory=True)

print("dataset loaded", flush=True)

model = Pangolin(L, W, AR)
if torch.cuda.device_count() > 1:
    print("Using %s gpus" % torch.cuda.device_count(), flush=True)
    model = nn.DataParallel(model)
model.cuda()

def crossent(y_pred, y_true):
    # Standard categorical cross entropy for sequence outputs
    return - torch.mean(y_true[:, 0, :]*torch.log(y_pred[:, 0, :]+1e-10)
                      + y_true[:, 1, :]*torch.log(y_pred[:, 1, :]+1e-10))

bce = nn.BCELoss()

def loss(y_pred, y_true):
    y_pred = torch.split(y_pred, [2,1,2,1,2,1,2,1], dim=1)
    y_true = torch.split(y_true, [2,1,2,1,2,1,2,1], dim=1)

    loss_cat = crossent(y_pred[0], y_true[0]) + crossent(y_pred[2], y_true[2]) + crossent(y_pred[4], y_true[4]) + crossent(y_pred[6], y_true[6])
    loss_cont = (bce(y_pred[1][y_true[1]>=0], y_true[1][y_true[1]>=0]) + bce(y_pred[3][y_true[3]>=0], y_true[3][y_true[3]>=0]) +
                 bce(y_pred[5][y_true[5]>=0], y_true[5][y_true[5]>=0]) + bce(y_pred[7][y_true[7]>=0], y_true[7][y_true[7]>=0]))

    return loss_cat + loss_cont

def train(epoch):
    print('\nEpoch: %d' % epoch, flush=True)
    model.train()
    train_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss += float(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + batch_idx / iters)

        #print(batch_idx, len(train_dl), 'Loss: %.5f' % (train_loss/(batch_idx+1)), flush=True)
        progress_bar(batch_idx, len(train_dl),
                     'Loss: %.5f' % (train_loss/(batch_idx+1)))

    return train_loss/batch_idx

def test():
    model.eval()
    test_loss = 0

    for batch_idx, (inputs, targets) in enumerate(val_dl):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        test_loss += float(loss)
        
        #print(batch_idx, len(val_dl), 'Loss: %.5f' % (test_loss/(batch_idx+1)), flush=True)
        progress_bar(batch_idx, len(val_dl),
                     'Loss: %.5f' % (test_loss/(batch_idx+1)))

    return test_loss/batch_idx

criterion = loss
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
# 2 4
T_0 = 2
T_mult = 2
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
iters = len(train_dl)
flog_final = open("log.%s.txt" % (model_num), 'w')

for epoch in range(0, 14):
    train_loss = train(epoch)
    test_loss = test()
    print(epoch, train_loss, test_loss, file=flog_final, flush=True)
    flog_final.flush()
    torch.save(model.state_dict(), "models/model.%s.%s" % (model_num, epoch))



