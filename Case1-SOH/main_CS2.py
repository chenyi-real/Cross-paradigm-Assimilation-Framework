# NASA main
from dataloader.dataloader import XJTUdata,MITdata,HUSTdata,TJUdata,NASAdata
from Model.Model import PINN, PINN_v2, PINN_v3, PINN_v4, PINN_v5, PINN_debug
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for MIT dataset')
    parser.add_argument('--data', type=str, default='MIT', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate of F')

    # model related
    parser.add_argument('--u_layers_num', type=int, default=3, help='the layers num of u')
    parser.add_argument('--u_hidden_dim', type=int, default=60, help='the hidden dim of u')
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=1, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.02, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='new_results/results of reviewer/NASA results', help='save folder')

    args = parser.parse_args()

    return args


def load_CS2_data(args,normalization=True,small_sample=None,test_id='CS2_35'):
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

def old_load_CS2_data(args,normalization=True,small_sample=None,test_id='CS2_35'):
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


def main(test_id='CS2_35'):
    args = get_args()
    for e in range(10):
        setattr(args, 'save_folder', f'new_results_v10/results of PINNv2/CS2 results-{test_id}/Experiment{e + 1}')
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)

        dataloader = old_load_CS2_data(args, normalization=True, test_id=test_id)
        pinn = PINN_v2(args)
        pinn.Train(trainloader=dataloader['train'],validloader=dataloader['valid'],testloader=dataloader['test'])

# def small_sample():
#     args = get_args()
#     num_battery = 2
#     for e in range(10):
#         setattr(args, 'save_folder',
#                 f'results of reviewer/MIT results (small sample {num_battery})/Experiment{e + 1}')
#         setattr(args, 'batch_size', 128)
#         if not os.path.exists(args.save_folder):
#             os.makedirs(args.save_folder)
#         dataloader = load_MIT_data(args, small_sample=num_battery)
#         pinn = PINN(args)
#         pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])


if __name__ == '__main__':
    for item in ['CS2_35','CS2_36', 'CS2_37', 'CS2_38']:
        print('Start Train', item)
        main(item)
