import sys
sys.path.append("../train")
from model import *
from utils import *

if sys.argv[1] == "spliceai":
    spliceai = True
else:
    spliceai = False

if spliceai:
    from tensorflow.keras.models import load_model
    import tensorflow as tf
    # https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    SpliceAI = []
    for i in range(1,6):
        SpliceAI.append(load_model('../SpliceAI/spliceai/models/spliceai' + str(i) + '.h5'))
else:
    weights = [sys.argv[1]]

    models = []
    for i in range(0,len(weights)):
        model = Pangolin(L, W, AR)
        model.cuda()
        model.load_state_dict(torch.load(weights[i]), strict=True)
        model.eval()
        models.append(model)

ds = H5Dataset(sys.argv[2])
val = round(0.1*len(ds))
train = len(ds)-val
train_ds, val_ds = data.random_split(ds, (train,val))
train_dl = data.DataLoader(train_ds, batch_size=1)
val_dl = data.DataLoader(ds, batch_size=6) #shuffle=True, num_workers=1, pin_memory=True)

#print(len(ds))
#dl = data.DataLoader(ds, batch_size=1)
#print(dl)

for i, x in enumerate(val_dl):
    print(i,x)

spliceai_outputs = np.empty([len(dl),3,5000], dtype=np.float16)
all_targets = np.empty([len(dl),12,5000], dtype=np.float16)
all_outputs = np.empty([len(dl),12,5000], dtype=np.float16) 

for batch_idx, (inputs, targets) in enumerate(dl):
    if batch_idx % 1000 == 0:
        print(batch_idx, flush=True)
    if batch_idx > 1000:
        break    
    all_targets[batch_idx:batch_idx+1,:,:] = targets.numpy()

    if spliceai:
        for model in SpliceAI:
            outputs = model.predict(inputs.permute(0,2,1).numpy(), batch_size=1)
            spliceai_outputs[batch_idx:batch_idx+1,:,:] += np.transpose(outputs,(0,2,1))
    else:
        inputs = inputs.cuda()
        all_targets[batch_idx:batch_idx+1,:,:] = targets.numpy()
        for model in models:
            all_outputs[batch_idx:batch_idx+1,:,:] += model(inputs).cpu().detach().numpy()

all_targets = np.split(all_targets, [2,3,5,6,8,9,11], axis=1)
if spliceai:
    print_metrics(all_targets[0], spliceai_outputs, spliceai=True)
    print_metrics(all_targets[2], spliceai_outputs, spliceai=True)
    print_metrics(all_targets[4], spliceai_outputs, spliceai=True)
    print_metrics(all_targets[6], spliceai_outputs, spliceai=True)
else:
    all_outputs = np.split(all_outputs, [2,3,5,6,8,9,11], axis=1)
    print_metrics(all_targets[0], all_outputs[0])
    print_metrics(all_targets[2], all_outputs[0])
    print_metrics(all_targets[4], all_outputs[0])
    print_metrics(all_targets[6], all_outputs[0])
