import torch
import torch.nn as nn
import numpy as np
import os
from utils.util import AverageMeter,get_logger
from Model.Compare_Models import MLP,CNN,Spikeformer,SpikeGRU,LSTM,TCN
from Model.Model import LR_Scheduler
from dataloader.dataloader import XJTUdata,HUSTdata,MITdata,TJUdata,NASAdata
import argparse

class Trainer():
    def __init__(self,model,train_loader,valid_loader,test_loader,args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.save_dir = args.save_folder
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.epochs = args.epochs
        self.logger = get_logger(os.path.join(args.save_folder,args.log_dir))


        self.loss_meter = AverageMeter()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.warmup_lr)
        self.scheduler = LR_Scheduler(optimizer=self.optimizer,
                                      warmup_epochs=args.warmup_epochs,
                                      warmup_lr=args.warmup_lr,
                                      num_epochs=args.epochs,
                                      base_lr=args.lr,
                                      final_lr=args.final_lr)
        self.best_model=None

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def train_one_epoch(self,epoch):
        self.model.train()
        self.loss_meter.reset()
        for (x1,_,y1,_) in self.train_loader:
            x1 = x1.to(self.device)
            y1 = y1.to(self.device)


            y_pred = self.model(x1)
            loss = self.loss_func(y_pred,y1)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_meter.update(loss.item())
            
        info = '[Train] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def valid(self,epoch):
        self.model.eval()
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1,_,y1,_) in self.valid_loader:
                x1 = x1.to(self.device)
                y1 = y1.to(self.device)

                y_pred = self.model(x1)
                loss = self.loss_func(y_pred,y1)
                self.loss_meter.update(loss.item())
        info = '[Valid] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def test(self):
        self.model.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x1,_,y1,_) in self.test_loader:
                x1 = x1.to(self.device)
                y_pred = self.model(x1)

                true_label.append(y1.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
        true_label = np.concatenate(true_label,axis=0)
        pred_label = np.concatenate(pred_label,axis=0)
        self.best_model = self.model.state_dict()
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir,'true_label.npy'),true_label)
            np.save(os.path.join(self.save_dir,'pred_label.npy'),pred_label)
        return true_label,pred_label

    def train(self):
        min_loss = 100
        early_stop = 0
        for epoch in range(1,self.epochs+1):
            early_stop += 1
            train_loss = self.train_one_epoch(epoch)
            current_lr = self.scheduler.step()
            valid_loss = self.valid(epoch)
            if valid_loss < min_loss and self.test_loader is not None:
                min_loss = valid_loss
                true_label,pred_label = self.test()
                early_stop = 0
            if early_stop > 10:
                break
        self.clear_logger()
        if self.save_dir is not None:
            torch.save(self.best_model,os.path.join(self.save_dir,'model.pth'))

def load_model(args):
    if args.model == 'MLP':
        model = MLP()
    elif args.model == 'CNN':
        model = CNN()
    elif args.model == 'Spikeformer':
        model = Spikeformer()
    elif args.model == 'SpikeGRU':
        model = SpikeGRU()
    elif args.model == 'LSTM':
        model = LSTM()
    elif args.model == 'TCN':
        model = TCN()
    return model

def load_dataset(args, test_id=None):
    if args.dataset == 'NASA':
        dataset = load_NASA_data(args, normalization=True, batch=test_id)
    elif args.dataset == 'MIT':
        dataset = load_MIT_data(args, normalization=True)
    elif args.dataset == 'CS2':
        dataset = new_load_CS2_data(args, normalization=True, test_id=test_id)
    elif args.dataset == 'MIX1':
        dataset = load_MIX_data(args, normalization=True, train_data=['CS2', 'MIT'], test_data='NASA', test_id=test_id)
    elif args.dataset == 'MIX2':
        dataset = load_MIX_data(args, normalization=True, train_data=['NASA', 'MIT'], test_data='CS2', test_id=test_id)
    elif args.dataset == 'MIX3':
        dataset = load_MIX_data(args, normalization=True, train_data=['NASA', 'CS2'], test_data='MIT', test_id=test_id)
    else:
        print('没有这种类型的数据集')
        exit()
    return dataset


def load_MIX_data(args,normalization=True,train_data=['CS2', 'MIT'], test_data='NASA', test_id=None):
    root_map = {'MIT':'hyx_data/MIT/HIs',
                'CS2':'hyx_data/CALCE/new_out/',
                'NASA':'hyx_data/NASA/new_out/'}
    
    train_list = []
    for data_id in train_data:
        root = root_map[data_id]
        if data_id == 'MIT':
            for batch in ['2017-05-12','2017-06-30']:
                batch_root = os.path.join(root,batch)
                files = os.listdir(batch_root)
                for f in files:
                    train_list.append(os.path.join(batch_root,f))
        elif data_id == 'CS2':
            for batch in ['CS2_35','CS2_36', 'CS2_37', 'CS2_38']:
                batch_root = os.path.join(root,f'{batch}_HIs_sorted.csv')
                train_list.append(batch_root)
        elif data_id == 'NASA':
            for batch in ['B0005','B0006', 'B0007', 'B0018']:
                batch_root = os.path.join(root,f'{batch}.csv')
                train_list.append(batch_root)
    

    test_list = []
    root = root_map[test_data]
    if test_data == 'MIT':
            for batch in ['2017-05-12','2017-06-30']:
                batch_root = os.path.join(root,batch)
                files = os.listdir(batch_root)
                for f in files:
                    test_list.append(os.path.join(batch_root,f))
    elif test_data == 'CS2':
        for batch in ['CS2_35','CS2_36', 'CS2_37', 'CS2_38']:
            if test_id == batch:
                batch_root = os.path.join(root,f'{batch}_HIs_sorted.csv')
                test_list.append(batch_root)
    elif test_data == 'NASA':
        for batch in ['B0005','B0006', 'B0007', 'B0018']:
            if test_id == batch:
                batch_root = os.path.join(root,f'{batch}.csv')
                test_list.append(batch_root)


    data = MITdata(root=root,args=args, normalization=normalization)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader


def load_XJTU_data(args,small_sample=None):
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for file in files:
        if args.xjtu_batch in file:
            if '4' in file or '8' in file:
                test_list.append(os.path.join(root, file))
            else:
                train_list.append(os.path.join(root, file))
    if small_sample is not None:
        train_list = train_list[:small_sample]

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
    return dataloader

def load_MIT_data(args,normalization=True,small_sample=None):
    root = 'hyx_data/MIT/HIs'
    # root = 'data/MIT data'
    train_list = []
    test_list = []
    for batch in ['2017-05-12','2017-06-30']:
        batch_root = os.path.join(root,batch)
        files = os.listdir(batch_root)
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root,f))
            else:
                train_list.append(os.path.join(batch_root,f))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    data = MITdata(root=root,args=args, normalization=normalization)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader


def load_NASA_data(args,normalization=True,small_sample=None, batch=None):
    test_id = [batch]
    root = 'hyx_data/NASA/new_out/'
    # root = 'data/MIT data'
    test_list = []

    for batch in test_id:
        batch_root = os.path.join(root,f'{batch}.csv')
        test_list.append(batch_root)
        
    data = NASAdata(root=root,args=args, normalization=normalization)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':testloader['train_2'],'valid':testloader['valid_2'],'test':testloader['valid_2']}



    return dataloader

# old version
def old_load_NASA_data(args,normalization=True,small_sample=None, batch='B0005'):
    test_id = [batch]
    batch_id = ['B0005','B0006', 'B0007', 'B0018']
    root = 'hyx_data/NASA/new_out/'
    # root = 'data/MIT data'
    train_list = []
    test_list = []

    for batch in batch_id:
        batch_root = os.path.join(root,f'{batch}.csv')
        if batch in test_id:
            test_list.append(batch_root)
        else:
            train_list.append(batch_root)
        
    if small_sample is not None:
        train_list = train_list[:small_sample]
        
    data = NASAdata(root=root,args=args, normalization=normalization)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader

def load_CS2_data(args,normalization=True,small_sample=None,test_id='CS2_35'):
    test_id = [test_id]
    batch_id = ['CS2_35','CS2_36', 'CS2_37', 'CS2_38']
    root = 'hyx_data/CALCE/new_out/'
    # root = 'data/MIT data'
    train_list = []
    test_list = []

    for batch in batch_id:
        batch_root = os.path.join(root,f'{batch}_HIs_sorted.csv')
        if batch in test_id:
            test_list.append(batch_root)
        else:
            train_list.append(batch_root)
        
    if small_sample is not None:
        train_list = train_list[:small_sample]
        
    data = NASAdata(root=root,args=args, normalization=normalization)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],'valid':trainloader['valid_2'],'test':testloader['test_3']}

    return dataloader


def new_load_CS2_data(args,normalization=True,small_sample=None,test_id='CS2_35'):
    root = 'hyx_data/CALCE/new_out/'
    # root = 'data/MIT data'
    test_list = []

    
    batch_root = os.path.join(root,f'{test_id}_HIs_sorted.csv')
    
    test_list.append(batch_root)
    
        
    if small_sample is not None:
        train_list = train_list[:small_sample]
        
    data = NASAdata(root=root,args=args, normalization=normalization)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':testloader['train_2'],'valid':testloader['valid_2'],'test':testloader['valid_2']}

    return dataloader

def get_args():
    parser = argparse.ArgumentParser('The parameters of Comparision methods')
    parser.add_argument('--model',type=str,default='CNN',choices=['MLP','CNN'])
    parser.add_argument('--dataset',type=str,default='MIT',choices=['XJTU','HUST','MIT','TJU'])
    parser.add_argument('--normalization_method',type=str, default='min-max', help='min-max,z-score')

    # XJTU data
    parser.add_argument('--xjtu_batch',type=str,default='2C',choices=['2C','3C','R2.5','R3','RW','satellite'])

    # TJU data
    parser.add_argument('--in_same_batch',type=bool,default=True)
    parser.add_argument('--tju_batch',type=int,default=0,choices=[0,1,2])
    parser.add_argument('--tju_train_batch',type=int,default=-1, choices=[-1,0,1,2])
    parser.add_argument('--tju_test_batch',type=int,default=-1, choices=[-1,0,1,2])

    # scheduler related
    parser.add_argument('--epochs', type=int, default=20, help='epoch')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-5, help='final lr')
    parser.add_argument('--lr_F', type=float, default=5e-5, help='lr of F')


    parser.add_argument('--save_folder',type=str,default='./results of reviewer/')
    parser.add_argument('--log_dir',type=str,default='logging.txt')
    parser.add_argument('--batch_size',type=int,default=32)

    args = parser.parse_args()
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    return args




def main(model='MLP', dataset='NASA', test_id=None):
    setattr(args,'model',model) # select model: MLP or CNN
    setattr(args,'dataset',dataset) # select model: MLP or CNN
    test_id = test_id
    print(f'Start Train {model} in {dataset} with {test_id}')
    for e in range(10):
        setattr(args,'save_folder',os.path.join('./new_results_10/',f'{args.dataset}-{args.model} results-{test_id}/Experiment{e+1}'))
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        model = load_model(args)
        data_loader = load_dataset(args, test_id=test_id)
        trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
        trainer.train()

def main_NASA(args, model='MLP', dataset='NASA'):

    batch_id = ['B0005','B0006', 'B0007', 'B0018']
    
    setattr(args,'model',model) # select model: MLP or CNN
    setattr(args,'dataset',dataset) # select model: MLP or CNN
    for batch in batch_id:
        print(f'Start Train {model} in {dataset} with {batch}')
        for e in range(10):
            setattr(args,'save_folder',os.path.join('./new_results_v9/',f'{args.dataset}-{args.model} results/{batch}/Experiment{e+1}'))
            if not os.path.exists(args.save_folder):
                os.makedirs(args.save_folder)

            model = load_model(args)
            data_loader = load_dataset(args, test_id=batch)
            trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
            trainer.train()


if __name__ == '__main__':
    args = get_args()

    # main_NASA(args,model='MLP')

    index = 9
   
    if index == 1:
        main('CNN', dataset='NASA', test_id='B0005')
        main('CNN', dataset='NASA', test_id='B0006')
        main('CNN', dataset='NASA', test_id='B0007')
        main('CNN', dataset='NASA', test_id='B0018')
    elif index == 2:
        main('SpikeGRU', dataset='NASA', test_id='B0005')
        main('SpikeGRU', dataset='NASA', test_id='B0006')
        main('SpikeGRU', dataset='NASA', test_id='B0007')
        main('SpikeGRU', dataset='NASA', test_id='B0018')
    elif index == 3:
        main('CNN', dataset='CS2', test_id='CS2_35')
        main('CNN', dataset='CS2', test_id='CS2_36')
        main('CNN', dataset='CS2', test_id='CS2_37')
        main('CNN', dataset='CS2', test_id='CS2_38')
    elif index == 4:
        main('CNN', dataset='MIT')
    elif index == 5:
        main('SpikeGRU', dataset='CS2', test_id='CS2_35')
        main('SpikeGRU', dataset='CS2', test_id='CS2_36')
        main('SpikeGRU', dataset='CS2', test_id='CS2_37')
        main('SpikeGRU', dataset='CS2', test_id='CS2_38')
    elif index == 6:
        main('SpikeGRU', dataset='MIT')
    elif index == 7:
        main('MLP', dataset='CS2', test_id='CS2_35')
        main('MLP', dataset='CS2', test_id='CS2_36')
        main('MLP', dataset='CS2', test_id='CS2_37')
        main('MLP', dataset='CS2', test_id='CS2_38')
    elif index == 8:
        

        main('LSTM', dataset='NASA', test_id='B0005')
        main('LSTM', dataset='NASA', test_id='B0006')
        main('LSTM', dataset='NASA', test_id='B0007')
        main('LSTM', dataset='NASA', test_id='B0018')

        main('LSTM', dataset='CS2', test_id='CS2_35')
        main('LSTM', dataset='CS2', test_id='CS2_36')
        main('LSTM', dataset='CS2', test_id='CS2_37')
        main('LSTM', dataset='CS2', test_id='CS2_38')

        main('LSTM', dataset='MIT')

    elif index == 9:

        main('TCN', dataset='NASA', test_id='B0005')
        main('TCN', dataset='NASA', test_id='B0006')
        main('TCN', dataset='NASA', test_id='B0007')
        main('TCN', dataset='NASA', test_id='B0018')

        main('TCN', dataset='CS2', test_id='CS2_35')
        main('TCN', dataset='CS2', test_id='CS2_36')
        main('TCN', dataset='CS2', test_id='CS2_37')
        main('TCN', dataset='CS2', test_id='CS2_38')

        main('TCN', dataset='MIT')


    # if index == 1:
    #      # MIX1
    #     main('MLP', dataset='MIX1', test_id='B0005')
    #     main('MLP', dataset='MIX1', test_id='B0006')
    #     main('MLP', dataset='MIX1', test_id='B0007')
    #     main('MLP', dataset='MIX1', test_id='B0018')

    #     main('Spikeformer', dataset='MIX1', test_id='B0005')
    #     main('Spikeformer', dataset='MIX1', test_id='B0006')
    #     main('Spikeformer', dataset='MIX1', test_id='B0007')
    #     main('Spikeformer', dataset='MIX1', test_id='B0018')
    # elif index == 2:
    #     # MIX2  
    #     main('MLP', dataset='MIX2', test_id='CS2_35')
    #     main('MLP', dataset='MIX2', test_id='CS2_36')
    #     main('MLP', dataset='MIX2', test_id='CS2_37')
    #     main('MLP', dataset='MIX2', test_id='CS2_38')

    #     main('Spikeformer', dataset='MIX2', test_id='CS2_35')
    #     main('Spikeformer', dataset='MIX2', test_id='CS2_36')
    #     main('Spikeformer', dataset='MIX2', test_id='CS2_37')
    #     main('Spikeformer', dataset='MIX2', test_id='CS2_38')

    # elif index == 3:
    #     # MIX3
    #     main('MLP', dataset='MIX3', test_id=None)
    #     main('MLP', dataset='MIX3', test_id=None)
    #     main('MLP', dataset='MIX3', test_id=None)
    #     main('MLP', dataset='MIX3', test_id=None)

    #     main('Spikeformer', dataset='MIX3', test_id=None)
    #     main('Spikeformer', dataset='MIX3', test_id=None)
    #     main('Spikeformer', dataset='MIX3', test_id=None)
    #     main('Spikeformer', dataset='MIX3', test_id=None)




    # main('MLP', dataset='CS2', test_id='CS2_35')
    # main('MLP', dataset='CS2', test_id='CS2_36')
    # main('MLP', dataset='CS2', test_id='CS2_37')
    # main('MLP', dataset='CS2', test_id='CS2_38')

    # main('Spikeformer', dataset='CS2', test_id='CS2_35')
    # main('Spikeformer', dataset='CS2', test_id='CS2_36')
    # main('Spikeformer', dataset='CS2', test_id='CS2_37')
    # main('Spikeformer', dataset='CS2', test_id='CS2_38')

    # # mlp
    # setattr(args,'model','MLP') # select model: MLP or CNN
    # setattr(args,'dataset','NASA') # select model: MLP or CNN
    # test_id = 'B0005'
    # for e in range(10):
    #     setattr(args,'save_folder',os.path.join('./new_results_v7/',f'{args.dataset}-{args.model} results-{test_id}/Experiment{e+1}'))
    #     if not os.path.exists(args.save_folder):
    #         os.makedirs(args.save_folder)

    #     model = load_model(args)
    #     data_loader = load_NASA_data(args, normalization=True, test_id=test_id)
    #     trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
    #     trainer.train()    


    # # mlp
    # setattr(args,'model','MLP') # select model: MLP or CNN
    # setattr(args,'dataset','NASA') # select model: MLP or CNN
    # test_id = 'B0006'
    # for e in range(10):
    #     setattr(args,'save_folder',os.path.join('./new_results_v7/',f'{args.dataset}-{args.model} results-{test_id}/Experiment{e+1}'))
    #     if not os.path.exists(args.save_folder):
    #         os.makedirs(args.save_folder)

    #     model = load_model(args)
    #     data_loader = load_NASA_data(args, normalization=True, test_id=test_id)
    #     trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
    #     trainer.train()  

    # # mlp
    # setattr(args,'model','MLP') # select model: MLP or CNN
    # setattr(args,'dataset','NASA') # select model: MLP or CNN
    # test_id = 'B0007'
    # for e in range(10):
    #     setattr(args,'save_folder',os.path.join('./new_results_v7/',f'{args.dataset}-{args.model} results-{test_id}/Experiment{e+1}'))
    #     if not os.path.exists(args.save_folder):
    #         os.makedirs(args.save_folder)

    #     model = load_model(args)
    #     data_loader = load_NASA_data(args, normalization=True, test_id=test_id)
    #     trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
    #     trainer.train()  

    #     # mlp
    # setattr(args,'model','MLP') # select model: MLP or CNN
    # setattr(args,'dataset','NASA') # select model: MLP or CNN
    # test_id = 'B0018'
    # for e in range(10):
    #     setattr(args,'save_folder',os.path.join('./new_results_v7/',f'{args.dataset}-{args.model} results-{test_id}/Experiment{e+1}'))
    #     if not os.path.exists(args.save_folder):
    #         os.makedirs(args.save_folder)

    #     model = load_model(args)
    #     data_loader = load_NASA_data(args, normalization=True, test_id=test_id)
    #     trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
    #     trainer.train()  


    

    # # spikeformer
    # setattr(args,'model','Spikeformer') # select model: MLP or CNN
    # setattr(args,'dataset','NASA') # select model: MLP or CNN
    # test_id = 'B0005'
    # for e in range(10):
    #     setattr(args,'save_folder',os.path.join('./new_results_v7/',f'{args.dataset}-{args.model} results-{test_id}/Experiment{e+1}'))
    #     if not os.path.exists(args.save_folder):
    #         os.makedirs(args.save_folder)

    #     model = load_model(args)
    #     data_loader = load_NASA_data(args, normalization=True, test_id=test_id)
    #     trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
    #     trainer.train()

    
    # setattr(args,'model','Spikeformer') # select model: MLP or CNN
    # setattr(args,'dataset','NASA') # select model: MLP or CNN
    # test_id = 'B0006'
    # for e in range(10):
    #     setattr(args,'save_folder',os.path.join('./new_results_v7/',f'{args.dataset}-{args.model} results-{test_id}/Experiment{e+1}'))
    #     if not os.path.exists(args.save_folder):
    #         os.makedirs(args.save_folder)

    #     model = load_model(args)
    #     data_loader = load_NASA_data(args, normalization=True,test_id=test_id)
    #     trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
    #     trainer.train()


    # setattr(args,'model','Spikeformer') # select model: MLP or CNN
    # setattr(args,'dataset','NASA') # select model: MLP or CNN
    # test_id = 'B0007'
    # for e in range(10):
    #     setattr(args,'save_folder',os.path.join('./new_results_v7/',f'{args.dataset}-{args.model} results-{test_id}/Experiment{e+1}'))
    #     if not os.path.exists(args.save_folder):
    #         os.makedirs(args.save_folder)

    #     model = load_model(args)
    #     data_loader = load_NASA_data(args, normalization=True,test_id=test_id)
    #     trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
    #     trainer.train()


    # setattr(args,'model','Spikeformer') # select model: MLP or CNN
    # setattr(args,'dataset','NASA') # select model: MLP or CNN
    # test_id = 'B0018'
    # for e in range(10):
    #     setattr(args,'save_folder',os.path.join('./new_results_v7/',f'{args.dataset}-{args.model} results-{test_id}/Experiment{e+1}'))
    #     if not os.path.exists(args.save_folder):
    #         os.makedirs(args.save_folder)

    #     model = load_model(args)
    #     data_loader = load_NASA_data(args, normalization=True,test_id=test_id)
    #     trainer = Trainer(model,data_loader['train'],data_loader['valid'],data_loader['test'],args)
    #     trainer.train()

