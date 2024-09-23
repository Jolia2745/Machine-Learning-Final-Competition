print("start")
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torchvision.transforms as transforms
import resnet
import data_loader as dl
import pandas as pd
from collections import OrderedDict
import numpy as np
import random
import librosa

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))
# batch size == learning rate 
print("finish importing")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)


class Args:
    def __init__(self):
        self.arch = 'resnet110'
        self.workers = 4
        self.epochs = 50
        self.start_epoch = 0
        self.batch_size = 32 # 128
        self.lr = 1e-4     # 0.01  # 
        self.momentum = 0.9
        self.weight_decay = 1e-2  # 1e-2       
        self.print_freq = 10
        self.evaluate = False
        self.pretrained = False 
        self.half = False
        self.save_dir = './resnet_model'
        self.save_every = 10



args = Args()
best_prec1 = 0




def random_time_shift(spectrogram):
    """shift wit time"""
    max_shift = spectrogram.shape[1] // 10  # at most 10% columns
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(spectrogram, shift, axis=1)

def add_random_noise(spectrogram, noise_level=0.02):
    """add random noise"""
    noise = np.random.randn(*spectrogram.shape) * noise_level
    #return spectrogram.cpu().numpy() + noise
    return spectrogram + noise

def random_pitch_shift(spectrogram, sr, n_steps):
    """change with pitch"""
    min_sr=8000
    max_sr=48000
    sr = np.random.randint(min_sr, max_sr + 1)
    # print(spectrogram, type(spectrogram)) tensor
    # spectrogram_np = np.asarray(spectrogram)
    return librosa.effects.pitch_shift(spectrogram, sr = sr, n_steps = n_steps)

def random_speed_change(spectrogram):
    """change with speed"""
    speed_factor = np.random.uniform(0.9, 1.1)
    spectrogram_speed = librosa.effects.time_stretch(spectrogram, rate = speed_factor)
    return spectrogram_speed

def apply_spec_augment(spectrogram, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    """SpecAugment: mask frequancy and time"""
    num_freq_bins = spectrogram.shape[0]
    num_time_bins = spectrogram.shape[1]

    for _ in range(num_mask):
        # mask frequency
        f = random.randint(0, int(num_freq_bins * freq_masking_max_percentage))
        f0 = random.randint(0, num_freq_bins - f)
        spectrogram[f0:f0+f, :] = 0

        # mask time
        t = random.randint(0, int(num_time_bins * time_masking_max_percentage))
        t0 = random.randint(0, num_time_bins - t)
        spectrogram[:, t0:t0+t] = 0


    return spectrogram

def apply_random_augmentation(spectrogram, sample_rate):
    # if random.random() < 0.5:
    #     spectrogram = random_time_shift(spectrogram)
    if random.random() < 0.5:
        spectrogram = add_random_noise(spectrogram)
    if random.random() < 0.5:
        n_steps = random.randint(-2, 2)  # stride of the change of pitch
        spectrogram = random_pitch_shift(spectrogram, sample_rate, n_steps)
    if random.random() < 0.5:
        spectrogram = random_speed_change(spectrogram)
    if random.random() < 0.5:
        spectrogram = apply_spec_augment(spectrogram)
    return spectrogram


def train_epoch(train_loader,  model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target_var = target.to(device)
        input_np = input.cpu().numpy()
        input_var = apply_random_augmentation(input_np, None)

        if args.half:
            input_var = input_var.half()

        # compute output
        # turn it into tensor
        input_var = torch.from_numpy(input_var)
        input_var = input_var.to(torch.float32)
        input_var = input_var.to(device)
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

       # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Train: Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    return losses.avg  

def validate(val_loader, model, criterion, scheduler):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input = input.to(device)

            if args.half:
                input = input.half()

            # Compute output
            output = model(input)
            loss = criterion(output, target)

            # Convert output to float if it's half precision
            output = output.float()
            loss = loss.float()

            # Measure accuracy and record loss
            prec1 = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(f'Valid: Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(f' * Prec@1 {top1.avg:.3f}')
    
    # Step the scheduler based on the average validation loss
    scheduler.step(losses.avg)
    print(f'Validation Loss: {losses.avg}')

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.th'):
    """
    Args:
        state (dict):  model optimizer / metadata
        is_best (bool)
        filename (str): file name of checkpoint
    """
    # save current checkpoint
    torch.save(state, filename)
    print("Checkpoint saved to {}".format(filename))

    # if current checkpoint is the best one, save it in another file
    if is_best:
        best_filename = os.path.splitext(filename)[0] + '_best.th'
        torch.save(state, best_filename)
        print("Best model saved to {}".format(best_filename))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def print_model_parameters(model):
    for name, param in model.named_parameters():
        print("{}: {}".format(name, param.size()))

def load_model(model, path):
    """
    Args:
        model (torch.nn.Module)
        checkpoint_path (str)
    
    Returns:
        model (torch.nn.Module)
    """
    model.eval()

    checkpoint = torch.load(path)
    new_state_dict = OrderedDict()

    # Adjusting keys if they start with 'module.' (saved from DataParallel)
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] if k.startswith('module.') else k  # strip off 'module.' if present
        new_state_dict[name] = v

    # Now load the adjusted state dict
    model.load_state_dict(new_state_dict)
    return model

def prediction(model, test_loader, prec):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            feature_batch = batch.to(device) # ! reminder: batch.size() : torch.Size([32, 1, 64, 259])    batch.size()[0] : torch.Size([1, 64, 259])
            # print(feature_batch.size()) # torch.Size([32, 1, 64, 259])
            # Add a channel dimension if it's missing
            if feature_batch.dim() == 3:
                 eature_batch = feature_batch.unsqueeze(1) 

            predictions_batch = model(feature_batch)

            _, predicted_classes = torch.max(predictions_batch, 1)
            all_predictions.extend(predicted_classes.cpu().tolist())  # ->cpu ->list

    submission = pd.DataFrame({
        'id': range(len(all_predictions)),  # create a range from 0 to len(all_predictions)-1
        'category': all_predictions  
    })

    submission.to_csv(f'./res_1_{prec}.csv', index=False)

def main():
    global args, best_prec1
    args = Args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = resnet.__dict__[args.arch]()

    model.to(device)

    cudnn.benchmark = True

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # print(normalize)
    print("Now begin to load dataset...")
    train_loader, val_loader, test_loader = dl.create_dataloaders('../data/train_images', '../data/test_images', '../data/train_label.txt',  batch_size=args.batch_size  , test_batch_size=args.batch_size)
    # define loss function (criterion) and optimizer
    
    
 
    criterion = nn.CrossEntropyLoss().to(device)
    if args.half:
        model.half()
        criterion.half()

    
    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    # #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                    # milestones=[10, 15, 30], last_epoch=args.start_epoch - 1) # milestone: lower the lr at 10 and 15 epoch
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)

    # # modify the learning rate
    # def lr_lambda(epoch):
    #     if epoch < 40:
    #         return 1  #1.0  # lr = 0.01
    #     elif 40 <= epoch < 50:
    #         return 0.1  # lr = 0.001
    #     else:
    #         return 0.01  # lr = 0.0001

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


    print("Now begin to train model...")
    for epoch in range(args.start_epoch, args.epochs):
        # if epoch == 0: # and i < 400:  for warm-up
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.lr * 0.1  # use low lr to stablize the model
        # else:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.lr  
        
        # train for one epoch
       
        loss_avg = train_epoch(train_loader, model, criterion, optimizer, epoch) # train: only train for 1 epoch
        # lr_scheduler.step()
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
   
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, lr_scheduler)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        min_loss_threshold = 0.07
        num_epochs_without_improvement = 0
        best_loss = float('inf')
        max_epochs_without_improvement = 5  # prevent the model fall into dead cycle
        
        
        # if  loss_avg  <= min_loss_threshold:
        #     print(f'Training stopped as average loss <= {min_loss_threshold}')
        #     break
        
        # Check if there's an improvement in the loss
        if loss_avg < best_loss:
            best_loss = loss_avg
            num_epochs_without_improvement = 0
        else:
            num_epochs_without_improvement += 1

        if is_best:
            prediction(model, test_loader,  prec1)

         # save model regularly
        if epoch > 0 and epoch % args.save_every == 0: 
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        # save the best checkpoint among all epochs
        save_checkpoint({ 
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

        
        # Check if training should be stopped due to no improvement
        if num_epochs_without_improvement >= max_epochs_without_improvement:
            print(f'Training stopped due to no improvement for {max_epochs_without_improvement} epochs')
            break


       
    # if args.evaluate:
    #     validate(val_loader, model, criterion, lr_scheduler)
    #     return

    print("Now begin to load model...")
    best_model_path = './resnet_model/model_best.th'
    model = load_model(model, best_model_path).to(device)
    # print_model_parameters(model)

   
    print("Now begin to predict labels for test set...")
    prediction(model, test_loader, None)



main()