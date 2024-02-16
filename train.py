from data.dataset import Synth90kDataset, synth90k_collate_fn
import torch.multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from models.crnn import CRNN, count_parameters
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm 
from models.ctc_decoder import ctc_decoder
from argparse import Namespace 



def getting_all_batches(batch, device):

    images, targets, target_lengths = batch['images'], \
                                      batch['targets'], \
                                    batch['target_lengths']
    images, targets, target_lengths = images.to(device), \
                                       targets.to(device), \
                                      target_lengths.to(device)
    return images, targets, target_lengths


def calculate_loss(preds, preds_length, targets, target_lengths, optimizer, criterion):
    optimizer.zero_grad()
    batch_size = images.size(0)
    
    loss = criterion(preds, targets, preds_length, target_lengths)

    
    loss.backward()

    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5) # gradient clipping with 5
    optimizer.step()
    return loss.item()
    

def calculate_accuracy(output, output_lengths, targets, target_lengths, 
                      decode_method = 'beam_search', beam_size = 10):
    output_detach = output.detach()
    preds = ctc_decoder(output_detach, method = decode_method, beam_size = beam_size)
    
    reals = targets.cpu().numpy().tolist()

    
    target_lengths = target_lengths.cpu().numpy().tolist()
    
    num_correct = 0
    target_length_counter = 0
    for pred, target_length in zip(preds, target_lengths):
        real = reals[target_length_counter: target_length_counter + target_length]
        target_length_counter += target_length

        # print(pred, real)
        if pred == real:
            num_correct += 1

    return num_correct


if __name__ == '__main__':
    # Set multiprocessing start method to 'spawn'
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_args = Namespace(
        data_dir = './data/mnt/ramdisk/max/90kDICT32px/',
        train_batch_size = 32,
        eval_batch_size = 512,
        epochs = 100,
        save_interval = 20,
        cpu_workers = 8,
        learning_rate = 0.05,
        reload_checkpoint = None,
        decode_method = 'beam_search',
        beam_size = 10,
        checkpoints_dir = 'checkpoints/',
        img_width = 100,
        img_height= 32,
        map_to_seq = 64,
        lstm_hidden = 256,
        leaky_relu = False
    )
    dataset_path = train_args.data_dir
    
    train_dataset = Synth90kDataset(dataset_path, mode = 'train', 
                                    img_height = train_args.img_height,
                                    img_width = train_args.img_width)
    valid_dataset = Synth90kDataset(dataset_path, mode = 'val', 
                                    img_height = train_args.img_height,
                                    img_width = train_args.img_width)

    reduced_train = len(train_dataset) // 25
    reduced_indices = torch.randperm(len(train_dataset))[:reduced_train]
    train_dataset_reduced = torch.utils.data.Subset(train_dataset, reduced_indices)
    
    reduced_val = len(valid_dataset) // 25
    reduced_indices = torch.randperm(len(train_dataset))[:reduced_val]
    val_dataset_reduced = torch.utils.data.Subset(valid_dataset, reduced_indices)
    
    mp.set_start_method('spawn', force=True)
    
    train_loader = DataLoader(train_dataset_reduced, batch_size = train_args.train_batch_size,
                             shuffle = True, num_workers = train_args.cpu_workers,
                            collate_fn = synth90k_collate_fn)
    valid_loader = DataLoader(val_dataset_reduced, batch_size = train_args.eval_batch_size,
                             shuffle = True, num_workers = train_args.cpu_workers,
                            collate_fn = synth90k_collate_fn)

    
    num_classes = len(Synth90kDataset.LABEL2CHAR) + 1
    crnn = CRNN(1, train_args.img_height, train_args.img_width, 
                num_classes = num_classes,
                leaky_relu = train_args.leaky_relu, 
                map_to_seq = train_args.map_to_seq,
                lstm_hidden = train_args.lstm_hidden).to(device)
    print(f"The number of parameters in this model are: {count_parameters(crnn)}")
    if train_args.reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    
    optimizer = optim.Adadelta(crnn.parameters(), lr = train_args.learning_rate, rho = 0.9)
    criterion = nn.CTCLoss(reduction = 'sum',  zero_infinity = True).to(device)
    
    num_epochs = train_args.epochs
    
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []
    
    for epoch in range(num_epochs):
        print(f"Running epoch: {epoch + 1}")
    
        crnn.train()
        running_loss = 0.0
        running_acc = 0.0
        total_loss = 0.0
        total_acc = 0.0
        total = 0
        train_bar = tqdm(desc = 'Training', total = len(train_loader),
                     position = 1, leave = True)
        
        for i, batch in enumerate(train_loader):
            
            images, targets, target_lengths = getting_all_batches(batch, device)
            batch_size = batch['images'].size(0)
    
            # print(images)
            
            preds = crnn(images)
            preds = preds.permute(1, 0, 2) #(seq_len, batch, num_classes)
            seq_length = preds.size(0)
    
            # print(preds)
            preds_lengths = torch.full(size = (batch_size, ), 
                                       fill_value = seq_length, 
                                       dtype = torch.long).to(device)

            loss_t = calculate_loss(preds, preds_lengths, targets, target_lengths,
                                   optimizer, criterion)
            
    
            running_loss += (loss_t - running_loss) / (i + 1)
            total_loss += loss_t 
            total += batch_size 
    
            
            num_correct = calculate_accuracy(preds, preds_lengths, 
                                              targets, target_lengths, 
                                             decode_method = train_args.decode_method,
                                             beam_size = train_args.beam_size)
            acc_t = num_correct / batch_size * 100
            running_acc += (acc_t - running_acc) / (i + 1)
            total_acc += num_correct 
            
            train_bar.set_postfix(loss = running_loss,
                                  acc = f"{running_acc:.2f}%",
                                  epoch = epoch + 1)
            train_bar.update()
    
        train_bar.close()
        
        current_loss = total_loss / len(train_loader)
        current_acc = total_acc / total * 100
        train_loss.append(current_loss)
        train_acc.append(current_acc)
    
        print("========================================")
        print("\033[1;34m" + f"Epoch {epoch + 1}/{num_epochs}" + "\033[0m")
        print(f"Train Loss: {current_loss:.2f}\nTrain Acc: {current_acc:.2f}%")

