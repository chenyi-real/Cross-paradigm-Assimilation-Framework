from Model.Model import PINN, PINN_v2, PINN_v3, PINN_debug, StudentMLP
from Model.Compare_Models import MLP,CNN,Spikeformer,SpikeGRU, LSTM, TCN
import torch
import torch.nn as nn
from dataloader.dataloader import XJTUdata,TJUdata, NASAdata, MITdata
from main_HUST import load_HUST_data
from main_MIT import load_MIT_data
from Model.Model import LR_Scheduler
import argparse
import os
import numpy as np
from utils.util import AverageMeter,eval_metrix,write_to_txt, get_logger

import torch.nn.functional as F


SOH_MAX = 0.9884384870529175
SOH_MIN = 0.9072180986404419



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def _l2_normalize(x, eps=1e-6):
    # x: [N, D]，按最后一维做 L2 归一化
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

###
class AdaPINN(PINN_debug):
    def __init__(self,args):
        super(AdaPINN, self).__init__(args)

        self.load_model(model_path=args.pretrain_model)
        self.ada_optimizer = torch.optim.Adam(self.solution_u.parameters(),lr=args.adaptation_lr)


    def adaptation_one_epoch(self,epoch,dataloader):
        self.solution_u.train()
        loss1_meter = AverageMeter()
        loss2_meter = AverageMeter()
        loss3_meter = AverageMeter()
        for iter,(x1,x2,y1,y2) in enumerate(dataloader):
            x1,x2,y1,y2 = x1.to(device),x2.to(device),y1.to(device),y2.to(device)
           
            u1,f1 = self.forward(x1)
            u2,f2 = self.forward(x2)

            # data loss
            loss1 = 0.5*self.loss_func(u1,y1) + 0.5*self.loss_func(u2,y2)

            # PDE loss
            f_target = torch.zeros_like(f1)
            loss2 = 0.5*self.loss_func(f1,f_target) + 0.5*self.loss_func(f2,f_target)

            # physics loss  u2-u1<0, considering capacity regeneration effect
            loss3 = self.relu(torch.mul(u2-u1,y1-y2)).sum()

            # total loss
            loss = loss1 + self.alpha*loss2 + self.beta*loss3

            self.ada_optimizer.zero_grad()
            loss.backward()
            self.ada_optimizer.step()

            loss1_meter.update(loss1.item())
            loss2_meter.update(loss2.item())
            loss3_meter.update(loss3.item())
            # debug_info = "[train] epoch:{} iter:{} data loss:{:.6f}, " \
            #              "PDE loss:{:.6f}, physics loss:{:.6f}, " \
            #              "total loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3,loss.item())
            # if epoch < 3:
            #     self.logger.debug(debug_info)

            if (iter+1) % 50 == 0:
                print("[epoch:{} iter:{}] data loss:{:.6f}, PDE loss:{:.6f}, physics loss:{:.6f}".format(epoch,iter+1,loss1,loss2,loss3))
        return loss1_meter.avg,loss2_meter.avg,loss3_meter.avg


    def Adaptation(self,trainloader,validloader=None,testloader=None):
        for param in self.dynamical_F.parameters(): # freeze the dynamical_F
            param.requires_grad = False

        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1, self.args.adaptation_epochs + 1):
            early_stop += 1
            loss1, loss2, loss3 = self.adaptation_one_epoch(e, trainloader)
            info = '[Train] epoch:{}, data loss:{:.6f}, ' \
                   'PDE loss:{:.6f}, ' \
                   'physics loss:{:.6f}, ' \
                   'total loss:{:.6f}'.format(e, loss1, loss2, loss3,
                                              loss1 + self.alpha * loss2 + self.beta * loss3)
            self.logger.info(info)
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                self.logger.info(info)
                early_stop = 0

                ############################### save ############################################
                self.best_model = {'solution_u': self.solution_u.state_dict(),
                                   'dynamical_F': self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
                ##################################################################################
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'finetune model.pth'))


    def _teacher_feature_forward(self, x):
        """
        Teacher 前向，返回:
        - u_t: teacher 的预测输出
        - feat_t: teacher 的特征（目前先用 u_t 代替）

        以后如果你在 PINN 里实现了真正的特征输出（比如
        forward_with_feat），只要改这里就行，蒸馏代码不用动。
        """
        u_t, feat_t = self.forward(x, return_feat=True)   # 你原有的接口：返回 (u, f)
        return u_t, feat_t


    def distill_one_epoch(
        self,
        epoch,
        student_model,
        optimizer,
        dataloader,
        hard_weight=1.0,
        feat_weight=1.0,
    ):
        """
        单个 epoch 的蒸馏训练：
        - Loss_hard: 学生输出 vs ground-truth
        - Loss_kd  : 学生特征 vs 教师特征（L2-normalize 后做 MSE）
        """
        # Teacher 固定
        self.solution_u.eval()
        self.dynamical_F.eval()

        # Student 训练
        student_model.train()

        hard_loss_meter = AverageMeter()
        kd_loss_meter = AverageMeter()

        for it, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2 = x1.to(device), x2.to(device)
            y1, y2 = y1.to(device), y2.to(device)

            # ---------- Teacher 前向（预测 + 特征） ----------
            with torch.no_grad():
                # 这里用我们前面约定的接口：
                # u_t, feat_t = self._teacher_feature_forward(x)
                u1_t, feat1_t = self._teacher_feature_forward(x1)
                u2_t, feat2_t = self._teacher_feature_forward(x2)

            # ---------- Student 前向（预测 + 特征） ----------
            # StudentMLP: forward(x) -> (u_s, feat_s)
            u1_s, feat1_s = student_model(x1)
            u2_s, feat2_s = student_model(x2)

            # ---------- 1) 硬标签损失（输出 vs GT） ----------
            hard_loss = 0.5 * self.loss_func(u1_s, y1) + \
                        0.5 * self.loss_func(u2_s, y2)

            # ---------- 2) 特征 KD 损失（L2 归一化后做 MSE） ----------
            # x1 上的特征对齐
            stu_feat1_n = _l2_normalize(feat1_s)
            tea_feat1_n = _l2_normalize(feat1_t.detach())

            # x2 上的特征对齐
            stu_feat2_n = _l2_normalize(feat2_s)
            tea_feat2_n = _l2_normalize(feat2_t.detach())

            kd_loss = 0.5 * F.mse_loss(stu_feat1_n, tea_feat1_n) + \
                    0.5 * F.mse_loss(stu_feat2_n, tea_feat2_n)

            # ---------- 总损失 ----------
            loss = hard_weight * hard_loss + feat_weight * kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hard_loss_meter.update(hard_loss.item())
            kd_loss_meter.update(kd_loss.item())

            if (it + 1) % 50 == 0:
                print(
                    "[Distill][epoch:{} iter:{}] "
                    "hard loss:{:.6f}, kd loss:{:.6f}, total loss:{:.6f}".format(
                        epoch, it + 1,
                        hard_loss.item(), kd_loss.item(), loss.item()
                    )
                )

        return hard_loss_meter.avg, kd_loss_meter.avg





    def Distill(
        self,
        student_model,
        trainloader,
        validloader=None,
        testloader=None,
        feat_loss_fn=None,
        hard_weight=None,
        feat_weight=None,
        num_epochs=None,
    ):
        """
        知识蒸馏主函数：
        - student_model: 学生 MLP（forward 返回 (u_s, feat_s)）
        - trainloader:   训练集
        - validloader:   验证集（可选）
        - testloader:    测试集（可选）
        """
        # ---------- 超参数 ----------
        if feat_loss_fn is None:
            feat_loss_fn = self.loss_func

        if hard_weight is None:
            hard_weight = getattr(self.args, "hard_weight", 1.0)

        if feat_weight is None:
            feat_weight = getattr(self.args, "feat_weight", 1.0)

        if num_epochs is None:
            num_epochs = getattr(
                self.args,
                "distill_epochs",
                getattr(self.args, "adaptation_epochs", 10)
            )

        distill_lr = getattr(self.args, "distill_lr", self.args.adaptation_lr)
        patience   = getattr(self.args, "distill_early_stop", None)

        # ---------- 冻结 Teacher ----------
        for p in self.solution_u.parameters():
            p.requires_grad = False
        for p in self.dynamical_F.parameters():
            p.requires_grad = False

        # ---------- Student 优化器 ----------
        distill_optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=distill_lr
        )

        best_student_state = None
        best_valid_mse = float("inf")
        early_stop = 0

        for e in range(1, num_epochs + 1):
            early_stop += 1

            hard_loss, kd_loss = self.distill_one_epoch(
                epoch=e,
                student_model=student_model,
                optimizer=distill_optimizer,
                dataloader=trainloader,
                hard_weight=hard_weight,
                feat_weight=feat_weight,
            )
            total_loss = hard_weight * hard_loss + feat_weight * kd_loss

            info = (
                f"[Distill-Train] epoch:{e}, "
                f"hard loss:{hard_loss:.6f}, "
                f"kd loss:{kd_loss:.6f}, "
                f"total loss:{total_loss:.6f}"
            )
            self.logger.info(info)

            # ---------- 验证 ----------
            if validloader is not None:
                valid_mse = self._evaluate_student_mse(student_model, validloader)
                self.logger.info(
                    f"[Distill-Valid] epoch:{e}, MSE:{valid_mse:.6f}"
                )

                if valid_mse < best_valid_mse:
                    best_valid_mse = valid_mse
                    early_stop = 0

                    # 测试集评估（可选）
                    if testloader is not None:
                        y_true, y_pred = self._collect_student_predictions(
                            student_model, testloader
                        )
                        MAE, MAPE, MSE, RMSE = eval_metrix(y_pred, y_true)
                        self.logger.info(
                            "[Distill-Test] MSE: {:.8f}, MAE: {:.6f}, "
                            "MAPE: {:.6f}, RMSE: {:.6f}".format(
                                MSE, MAE, MAPE, RMSE
                            )
                        )
                        ############################### save ############################################
                        if self.args.save_folder is not None:
                            np.save(os.path.join(self.args.save_folder, 'true_label.npy'), y_true)
                            np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), y_pred)
                        ##################################################################################

                    best_student_state = student_model.state_dict()

            # ---------- 早停 ----------
            if patience is not None and early_stop > patience:
                info = f"[Distill] early stop at epoch {e}"
                self.logger.info(info)
                break

        # ---------- 还原 best student 参数并保存 ----------
        if best_student_state is not None:
            student_model.load_state_dict(best_student_state)

        if self.args.save_folder is not None:
            torch.save(
                best_student_state if best_student_state is not None else student_model.state_dict(),
                os.path.join(self.args.save_folder, "student_distill_model.pth")
            )



    @torch.no_grad()
    def _evaluate_student_mse(self, student_model, dataloader):
        """
        学生模型在验证集上的 MSE 评估，
        dataloader: (x1, x2, y1, y2)
        """
        student_model.eval()
        mse_meter = AverageMeter()

        for x1, x2, y1, y2 in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            y1, y2 = y1.to(device), y2.to(device)

            u1_s, _ = student_model(x1)
            u2_s, _ = student_model(x2)

            mse = 0.5 * self.loss_func(u1_s, y1) + \
                  0.5 * self.loss_func(u2_s, y2)
            mse_meter.update(mse.item())

        return mse_meter.avg

    @torch.no_grad()
    def _collect_student_predictions(self, student_model, dataloader):
        """
        收集学生模型在测试集上的预测和真值，用于 eval_metrix。
        """
        student_model.eval()
        all_true = []
        all_pred = []

        for x1, x2, y1, y2 in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            y1, y2 = y1.to(device), y2.to(device)

            u1_s, _ = student_model(x1)
            u2_s, _ = student_model(x2)

            all_true.append(y1.detach().cpu())
            all_true.append(y2.detach().cpu())
            all_pred.append(u1_s.detach().cpu())
            all_pred.append(u2_s.detach().cpu())

        all_true = torch.cat(all_true, dim=0).numpy()
        all_pred = torch.cat(all_pred, dim=0).numpy()
        return all_true, all_pred



class AdaModel(CNN):
    def __init__(self,args):
        super(AdaModel, self).__init__()

        self.load_model(model_path=args.pretrain_model)
        self.ada_optimizer = torch.optim.Adam(self.parameters(),lr=args.adaptation_lr)
        self.loss_func = nn.MSELoss()
        self.loss_meter = AverageMeter()
        self.logger = get_logger(os.path.join(args.save_folder,args.log_dir))
        self.best_model=None
        self = self.to(device)
        # self.encoder = self.encoder.to(device)
        # self.predictor = self.predictor.to(device)
        self.save_dir = args.save_folder
        self.args = args

    def adaptation_one_epoch(self,epoch,dataloader):
        self.train()
       
        for (x1,_,y1,_) in dataloader:
            x1 = x1.to(device)
            y1 = y1.to(device)

            y_pred = self.forward(x1)
        
            loss = self.loss_func(y_pred,y1)
         

            self.ada_optimizer.zero_grad()
            loss.backward()
            self.ada_optimizer.step()

            self.loss_meter.update(loss.item())
            
            

        info = '[Train] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def Adaptation(self,trainloader,validloader=None,testloader=None):

        # for param in self.encoder.parameters(): # freeze the dynamical_F
        #     param.requires_grad = False

        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1, self.args.adaptation_epochs + 1):
            early_stop += 1
            loss = self.adaptation_one_epoch(e, trainloader)
            
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(e, validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                # [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                # info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                # self.logger.info(info)
                early_stop = 0

            if self.args.early_stop is not None and early_stop > 10:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'finetune model.pth'))


    def Valid(self, epoch, validloader):
        self.eval()
        
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1,_,y1,_) in validloader:
                x1 = x1.to(device)
                y1 = y1.to(device)

                y_pred = self.forward(x1)
                loss = self.loss_func(y_pred,y1)
                self.loss_meter.update(loss.item())
        info = '[Valid] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg
    
    def Test(self, testloader):
        self.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x1,_,y1,_) in testloader:
                x1 = x1.to(device)
                y_pred = self.forward(x1)

                true_label.append(y1.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
        true_label = np.concatenate(true_label,axis=0)
        pred_label = np.concatenate(pred_label,axis=0)
        self.best_model = self.state_dict()
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir,'true_label.npy'),true_label)
            np.save(os.path.join(self.save_dir,'pred_label.npy'),pred_label)
        return true_label,pred_label
    

class AdaMLPModel(MLP):
    def __init__(self,args):
        super(AdaMLPModel, self).__init__()

        self.load_model(model_path=args.pretrain_model)
        self.ada_optimizer = torch.optim.Adam(self.parameters(),lr=args.adaptation_lr)
        self.loss_func = nn.MSELoss()
        self.loss_meter = AverageMeter()
        self.logger = get_logger(os.path.join(args.save_folder,args.log_dir))
        self.best_model=None
        self = self.to(device)
        self.encoder = self.encoder.to(device)
        self.predictor = self.predictor.to(device)
        self.save_dir = args.save_folder
        self.args = args

    def adaptation_one_epoch(self,epoch,dataloader):
        self.train()
       
        for (x1,_,y1,_) in dataloader:
            x1 = x1.to(device)
            y1 = y1.to(device)

            y_pred = self.forward(x1)
        
            loss = self.loss_func(y_pred,y1)
         

            self.ada_optimizer.zero_grad()
            loss.backward()
            self.ada_optimizer.step()

            self.loss_meter.update(loss.item())
            
            

        info = '[Train] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def Adaptation(self,trainloader,validloader=None,testloader=None):

        for param in self.encoder.parameters(): # freeze the dynamical_F
            param.requires_grad = False
        for param in self.predictor.parameters(): # freeze the dynamical_F
            param.requires_grad = True

        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1, self.args.adaptation_epochs + 1):
            early_stop += 1
            loss = self.adaptation_one_epoch(e, trainloader)
            
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(e, validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                # [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                # info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                # self.logger.info(info)
                early_stop = 0

            if self.args.early_stop is not None and early_stop > 10:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'finetune model.pth'))


    def Valid(self, epoch, validloader):
        self.eval()
        
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1,_,y1,_) in validloader:
                x1 = x1.to(device)
                y1 = y1.to(device)

                y_pred = self.forward(x1)
                loss = self.loss_func(y_pred,y1)
                self.loss_meter.update(loss.item())
        info = '[Valid] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg
    
    def Test(self, testloader):
        self.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x1,_,y1,_) in testloader:
                x1 = x1.to(device)
                y_pred = self.forward(x1)

                true_label.append(y1.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
        true_label = np.concatenate(true_label,axis=0)
        pred_label = np.concatenate(pred_label,axis=0)
        self.best_model = self.state_dict()
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir,'true_label.npy'),true_label)
            np.save(os.path.join(self.save_dir,'pred_label.npy'),pred_label)
        return true_label,pred_label


class AdaCNNModel(CNN):
    def __init__(self,args):
        super(AdaCNNModel, self).__init__()

        self.load_model(model_path=args.pretrain_model)
        self.ada_optimizer = torch.optim.Adam(self.parameters(),lr=args.adaptation_lr)
        self.loss_func = nn.MSELoss()
        self.loss_meter = AverageMeter()
        self.logger = get_logger(os.path.join(args.save_folder,args.log_dir))
        self.best_model=None
        self = self.to(device)
        # self.encoder = self.encoder.to(device)
        # self.predictor = self.predictor.to(device)
        self.save_dir = args.save_folder
        self.args = args

    def adaptation_one_epoch(self,epoch,dataloader):
        self.train()
       
        for (x1,_,y1,_) in dataloader:
            x1 = x1.to(device)
            y1 = y1.to(device)

            y_pred = self.forward(x1)
        
            loss = self.loss_func(y_pred,y1)
         

            self.ada_optimizer.zero_grad()
            loss.backward()
            self.ada_optimizer.step()

            self.loss_meter.update(loss.item())
            
            

        info = '[Train] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def Adaptation(self,trainloader,validloader=None,testloader=None):
        # 1. Freeze all parameters first

        for name, param in self.named_parameters():
            param.requires_grad = False

        # 2. Unfreeze layer6
        for name, param in self.layer6.named_parameters():
            param.requires_grad = True

        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1, self.args.adaptation_epochs + 1):
            early_stop += 1
            loss = self.adaptation_one_epoch(e, trainloader)
            
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(e, validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                # [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                # info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                # self.logger.info(info)
                early_stop = 0

            if self.args.early_stop is not None and early_stop > 10:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'finetune model.pth'))


    def Valid(self, epoch, validloader):
        self.eval()
        
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1,_,y1,_) in validloader:
                x1 = x1.to(device)
                y1 = y1.to(device)

                y_pred = self.forward(x1)
                loss = self.loss_func(y_pred,y1)
                self.loss_meter.update(loss.item())
        info = '[Valid] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg
    
    def Test(self, testloader):
        self.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x1,_,y1,_) in testloader:
                x1 = x1.to(device)
                y_pred = self.forward(x1)

                true_label.append(y1.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
        true_label = np.concatenate(true_label,axis=0)
        pred_label = np.concatenate(pred_label,axis=0)
        self.best_model = self.state_dict()
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir,'true_label.npy'),true_label)
            np.save(os.path.join(self.save_dir,'pred_label.npy'),pred_label)
        return true_label,pred_label
    

class AdaLSTMModel(LSTM):
    def __init__(self,args):
        super(AdaLSTMModel, self).__init__()

        self.load_model(model_path=args.pretrain_model)
        self.ada_optimizer = torch.optim.Adam(self.parameters(),lr=args.adaptation_lr)
        self.loss_func = nn.MSELoss()
        self.loss_meter = AverageMeter()
        self.logger = get_logger(os.path.join(args.save_folder,args.log_dir))
        self.best_model=None
        self = self.to(device)
        # self.encoder = self.encoder.to(device)
        # self.predictor = self.predictor.to(device)
        self.save_dir = args.save_folder
        self.args = args

    def adaptation_one_epoch(self,epoch,dataloader):
        self.train()
       
        for (x1,_,y1,_) in dataloader:
            x1 = x1.to(device)
            y1 = y1.to(device)

            y_pred = self.forward(x1)
        
            loss = self.loss_func(y_pred,y1)
         

            self.ada_optimizer.zero_grad()
            loss.backward()
            self.ada_optimizer.step()

            self.loss_meter.update(loss.item())
            
            

        info = '[Train] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def Adaptation(self,trainloader,validloader=None,testloader=None):

        for param in self.lstm.parameters(): # freeze the dynamical_F
            param.requires_grad = False
        for param in self.fc.parameters(): # freeze the dynamical_F
            param.requires_grad = True

        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1, self.args.adaptation_epochs + 1):
            early_stop += 1
            loss = self.adaptation_one_epoch(e, trainloader)
            
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(e, validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                # [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                # info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                # self.logger.info(info)
                early_stop = 0

            if self.args.early_stop is not None and early_stop > 10:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'finetune model.pth'))


    def Valid(self, epoch, validloader):
        self.eval()
        
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1,_,y1,_) in validloader:
                x1 = x1.to(device)
                y1 = y1.to(device)

                y_pred = self.forward(x1)
                loss = self.loss_func(y_pred,y1)
                self.loss_meter.update(loss.item())
        info = '[Valid] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg
    
    def Test(self, testloader):
        self.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x1,_,y1,_) in testloader:
                x1 = x1.to(device)
                y_pred = self.forward(x1)

                true_label.append(y1.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
        true_label = np.concatenate(true_label,axis=0)
        pred_label = np.concatenate(pred_label,axis=0)
        self.best_model = self.state_dict()
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir,'true_label.npy'),true_label)
            np.save(os.path.join(self.save_dir,'pred_label.npy'),pred_label)
        return true_label,pred_label


class AdaTCNModel(TCN):
    def __init__(self,args):
        super(AdaTCNModel, self).__init__()

        self.load_model(model_path=args.pretrain_model)
        self.ada_optimizer = torch.optim.Adam(self.parameters(),lr=args.adaptation_lr)
        self.loss_func = nn.MSELoss()
        self.loss_meter = AverageMeter()
        self.logger = get_logger(os.path.join(args.save_folder,args.log_dir))
        self.best_model=None
        self = self.to(device)
        # self.encoder = self.encoder.to(device)
        # self.predictor = self.predictor.to(device)
        self.save_dir = args.save_folder
        self.args = args

    def adaptation_one_epoch(self,epoch,dataloader):
        self.train()
       
        for (x1,_,y1,_) in dataloader:
            x1 = x1.to(device)
            y1 = y1.to(device)

            y_pred = self.forward(x1)
        
            loss = self.loss_func(y_pred,y1)
         

            self.ada_optimizer.zero_grad()
            loss.backward()
            self.ada_optimizer.step()

            self.loss_meter.update(loss.item())
            
            

        info = '[Train] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg

    def clear_logger(self):
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.handlers.clear()

    def Adaptation(self,trainloader,validloader=None,testloader=None):

        for param in self.network.parameters(): # freeze the dynamical_F
            param.requires_grad = False
        for param in self.fc.parameters(): # freeze the dynamical_F
            param.requires_grad = True

        min_valid_mse = 10
        valid_mse = 10
        early_stop = 0
        mae = 10
        for e in range(1, self.args.adaptation_epochs + 1):
            early_stop += 1
            loss = self.adaptation_one_epoch(e, trainloader)
            
            if e % 1 == 0 and validloader is not None:
                valid_mse = self.Valid(e, validloader)
                info = '[Valid] epoch:{}, MSE: {}'.format(e, valid_mse)
                self.logger.info(info)
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                # [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)
                # info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
                # self.logger.info(info)
                early_stop = 0

            if self.args.early_stop is not None and early_stop > 10:
                info = 'early stop at epoch {}'.format(e)
                self.logger.info(info)
                break
        self.clear_logger()
        if self.args.save_folder is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'finetune model.pth'))


    def Valid(self, epoch, validloader):
        self.eval()
        
        self.loss_meter.reset()
        with torch.no_grad():
            for (x1,_,y1,_) in validloader:
                x1 = x1.to(device)
                y1 = y1.to(device)

                y_pred = self.forward(x1)
                loss = self.loss_func(y_pred,y1)
                self.loss_meter.update(loss.item())
        info = '[Valid] epoch:{:0>3d}, data loss:{:.6f}'.format(epoch,self.loss_meter.avg)
        self.logger.info(info)
        return self.loss_meter.avg
    
    def Test(self, testloader):
        self.eval()
        self.loss_meter.reset()
        true_label = []
        pred_label = []
        with torch.no_grad():
            for (x1,_,y1,_) in testloader:
                x1 = x1.to(device)
                y_pred = self.forward(x1)

                true_label.append(y1.cpu().detach().numpy())
                pred_label.append(y_pred.cpu().detach().numpy())
        true_label = np.concatenate(true_label,axis=0)
        pred_label = np.concatenate(pred_label,axis=0)

        self.best_model = self.state_dict()
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir,'true_label.npy'),true_label)
            np.save(os.path.join(self.save_dir,'pred_label.npy'),pred_label)
        return true_label,pred_label


def load_XJTU_data(args,small_sample=None):
    root = 'data/XJTU data'
    batch_names= ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
    batch_num = args.target_batch if args.target_data == 'XJTU' else args.source_batch
    batch = batch_names[batch_num]
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    for file in files:
        if batch in file:
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


def load_TJU_data(args,small_sample=None):
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []

    mod = [(5,9),(4,8),(5,9)]
    batchs = os.listdir(root)
    batch_num = args.target_batch if args.target_data == 'TJU' else args.source_batch
    batch = batchs[batch_num]
    batch_root = os.path.join(root,batch)
    files = os.listdir(batch_root)
    for i,f in enumerate(files):
        id = i + 1
        if id % 10 == mod[batch_num][0] or id % 10 == mod[batch_num][1]:
            test_list.append(os.path.join(batch_root,f))

        else:
            train_list.append(os.path.join(batch_root,f))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {'train': train_loader['train_2'],
                  'valid': train_loader['valid_2'],
                  'test': test_loader['test_3']}
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


def load_CS2_data(args,normalization=True,small_sample=None,batch='CS2_35'):
    test_id = [batch]
    root = 'hyx_data/CALCE/new_out/'
    # root = 'data/MIT data'
    test_list = []

    for batch in test_id:
        batch_root = os.path.join(root,f'{batch}_HIs_sorted.csv')
        
        test_list.append(batch_root)

        
        
    data = NASAdata(root=root,args=args, normalization=normalization)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':testloader['train_2'],'valid':testloader['valid_2'],'test':testloader['valid_2']}

    return dataloader


def load_UDDS_data(args,normalization=True,small_sample=None):
    root = 'hyx_data/UDDS/csv/'
    train_list = []
    test_list = []

    files = os.listdir(root)
    for file in files:
        # if 'C12' in file or 'C13' in file or 'C14' in file:
        if 'C14' in file:
            test_list.append(os.path.join(root, file))
        else:
            train_list.append(os.path.join(root, file))
    
        
    data = NASAdata(root=root,args=args, normalization=normalization)

    
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {'train':trainloader['train_2'],
                  'valid':trainloader['valid_2'],
                  'test':trainloader['valid_2']}

    return dataloader



def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=10, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup lr')
    parser.add_argument('--lr', type=float, default=0.01, help='base lr')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final lr')
    parser.add_argument('--lr_F', type=float, default=0.01, help='lr of F')

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.7, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='adaPINN_test', help='save folder')

    # The AdaPINN class inherits the PINN class, and the above parameters are all parameters of PINN.
    # The following are the parameters of AdaPINN.
    # adaption related
    parser.add_argument('--pretrain_model', type=str, default=None, help='The saving path of the model trained in the source domain')
    parser.add_argument('--adaptation_lr', type=float, default=1e-2, help='adaption lr')
    parser.add_argument('--adaptation_epochs', type=int, default=200, help='adaption epochs')
    parser.add_argument('--distill_early_stop', type=int, default=10)
    parser.add_argument('--target_data', type=str, default='XJTU', help='XJTU, HUST, MIT, TJU')
    parser.add_argument('--target_batch', type=int, default=-1, choices=[-1,0,1,2,3,4,5],
                        help='XJTU dataset is divided into 6 batches, and TJU dataset is divided into 3 batches. '
                             'If target_data is XJTU, the value range of target_batch is [-1,0,1,2,3,4,5];'
                             'If target_data is TJU, the value range of target_batch is [-1,0,1,2];'
                             'If it is other datasets, ignore target_batch')

    args = parser.parse_args()

    return args


def one_adaptation_task(args,source,target,source_batch=-1,target_batch=-1, model='PINN_v2'):
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if source in ['XJTU','TJU']:
        model_dir = f'./pretrained model/model_{source}_{source_batch}.pth'
    elif source in ['MIX1']:
        model_dir = f'/workspace/code/nc_proj/soh_pinn_v2/PINN4SOH-main/exp_20250903_table4/MIT-{model} results-None/Experiment1/model.pth'
    elif source in ['MIX2']:
        model_dir = f'/workspace/code/nc_proj/soh_pinn_v2/PINN4SOH-main/exp_20250903_table4/MIT-{model} results-None/Experiment1/model.pth'
    elif source in ['MIX3']:
        model_dir = f'./new_results_v8_table5/results of {model}/MIX3 results-None/Experiment1/model.pth'
    elif source in ['MIT']:
        model_dir = f'./exp_20251115_Transfer_base_MIT/MIT-{model} results/Experiment1/model.pth'
    elif source in ['UDDS']:
        model_dir = f'./exp_20251115_Transfer_MIT_UDDS_2/MIT-UDDS-{model}/Experiment1/finetune model.pth'
    else:
        model_dir = f'./pretrained model/model_{source}.pth'
    setattr(args,'pretrain_model',model_dir)
    setattr(args,'target_data',target)
    setattr(args,'target_batch',target_batch)

    # load data
    if target_batch == -1:
        target_loader = eval(f'load_{target}_data')(args,small_sample=None)
    else:
        target_loader = eval(f'load_{target}_data')(args,small_sample=None, batch=target_batch)

    # load model
    if model == 'Ours':
        model = AdaPINN(args)
    elif model == 'MLP':
        model = AdaMLPModel(args)
    elif model == 'CNN':
        model = AdaCNNModel(args)
    elif model == 'LSTM':
        model = AdaLSTMModel(args)
    elif model == 'TCN':
        model = AdaTCNModel(args)
    else:
        print('model is unrecognized')
        exit()

    # Firstly, test source model in target domain
    true_label,pred_label = model.Test(target_loader['test'])
    [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)

    print('Before adaptation (source only):')
    print('MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE))
    if args.log_dir is not None and args.save_folder is not None:
        save_name = os.path.join(args.save_folder,args.log_dir)
        info = 'Source only: {} -> {} | MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(source,target,MSE, MAE, MAPE, RMSE)
        write_to_txt(save_name,info)

    # adaptation
    model.Adaptation(trainloader=target_loader['train'],validloader=target_loader['valid'],testloader=target_loader['test'])




def one_distill_task(args,source,target,source_batch=-1,target_batch=-1, model='PINN_v2'):
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if source in ['XJTU','TJU']:
        model_dir = f'./pretrained model/model_{source}_{source_batch}.pth'
    elif source in ['MIX1']:
        model_dir = f'/workspace/code/nc_proj/soh_pinn_v2/PINN4SOH-main/exp_20250903_table4/MIT-{model} results-None/Experiment1/model.pth'
    elif source in ['MIX2']:
        model_dir = f'/workspace/code/nc_proj/soh_pinn_v2/PINN4SOH-main/exp_20250903_table4/MIT-{model} results-None/Experiment1/model.pth'
    elif source in ['MIX3']:
        model_dir = f'./new_results_v8_table5/results of {model}/MIX3 results-None/Experiment1/model.pth'
    elif source in ['MIT']:
        model_dir = f'./exp_20251115_Transfer_base_MIT/MIT-{model} results/Experiment1/model.pth'
    elif source in ['UDDS']:
        model_dir = f'./exp_20260101_Transfer_MIT_UDDS_2/MIT-UDDS-{model}/Experiment1/finetune model.pth'
    else:
        model_dir = f'./pretrained model/model_{source}.pth'

    setattr(args,'pretrain_model',model_dir)
    setattr(args,'target_data',target)
    setattr(args,'target_batch',target_batch)

    # load data
    if target_batch == -1:
        target_loader = eval(f'load_{target}_data')(args,small_sample=None)
    else:
        target_loader = eval(f'load_{target}_data')(args,small_sample=None, batch=target_batch)

    # load model
    if model == 'Ours':
        model = AdaPINN(args)
    elif model == 'MLP':
        model = AdaMLPModel(args)
    elif model == 'CNN':
        model = AdaCNNModel(args)
    elif model == 'LSTM':
        model = AdaLSTMModel(args)
    elif model == 'TCN':
        model = AdaTCNModel(args)
    else:
        print('model is unrecognized')
        exit()


    student_model = StudentMLP(input_dim=13, feature_dim=48).to(device)

    # Firstly, test source model in target domain
    true_label,pred_label = model.Test(target_loader['test'])
    [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)

    print('Before adaptation (source only):')
    print('MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE))
    if args.log_dir is not None and args.save_folder is not None:
        save_name = os.path.join(args.save_folder,args.log_dir)
        info = 'Source only: {} -> {} | MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(source,target,MSE, MAE, MAPE, RMSE)
        write_to_txt(save_name,info)

    # distill
    model.Distill(
        student_model=student_model,
        trainloader=target_loader['train'],
        validloader=target_loader['valid'],
        testloader=target_loader['test'],
        hard_weight=1,
        feat_weight=0.2,             # 你可以自己调：比如 0.5 或 2.0
    ) # type: ignore


def FineTune_TJU2XJTU():
    args = get_args()
    lrs = [0.0004,0.01,0.0005,0.002,0.003,0.0006]
    source_batchs = [2,2,2,1,0,1]
    target_batchs = [0,1,2,3,4,5]
    for lr,sb,tb in zip(lrs,source_batchs,target_batchs):
        for experiment in range(10):
            setattr(args,'adaptation_lr',lr)
            setattr(args,'log_dir','logging.txt')
            setattr(args,'save_folder',f'./results_fine-tuning/TJU-XJTU/batch{tb}/Experiment{experiment}')
            one_adaptation_task(args,source='TJU',target='XJTU',source_batch=sb,target_batch=tb)

def FineTune_XJTU2TJU():
    args = get_args()
    lrs = [0.003,0.002,0.002]
    source_batchs = [3,3,2]
    target_batchs = [0,1,2]
    for lr,sb,tb in zip(lrs,source_batchs,target_batchs):
        for experiment in range(10):
            setattr(args,'adaptation_lr',lr)
            setattr(args,'log_dir','logging.txt')
            setattr(args,'save_folder',f'./results_fine-tuning/XJTU-TJU/batch{tb}/Experiment{experiment}')
            one_adaptation_task(args,source='XJTU',target='TJU',source_batch=sb,target_batch=tb)


def FineTune_HUST2MIT():
    args = get_args()
    for experiment in range(10):
        setattr(args,'adaptation_lr',0.005)
        setattr(args,'log_dir','logging.txt')
        setattr(args,'save_folder',f'./results_fine-tuning/HUST-MIT/Experiment{experiment}')
        one_adaptation_task(args,source='HUST',target='MIT')

def FineTune_MIT2HUST():
    args = get_args()
    for experiment in range(10):
        setattr(args,'adaptation_lr',0.0002)
        setattr(args,'log_dir','logging.txt')
        setattr(args,'save_folder',f'./results_fine-tuning/MIT-HUST/Experiment{experiment}')
        one_adaptation_task(args,source='MIT',target='HUST')

# 分割线
def FineTune_MIX12NASA(model='PINN'):
    args = get_args()
    batch_id = ['B0005','B0006', 'B0007', 'B0018']
    for batch in batch_id:
        for experiment in range(10):
            setattr(args,'adaptation_lr',0.002) # 0.0002
            setattr(args,'log_dir','logging.txt')
            setattr(args,'save_folder',f'./results_fine-tuning-v4-model/MIX1-{model} results/{batch}/Experiment{experiment+1}')
            one_adaptation_task(args,source='MIX1',target='NASA',target_batch=batch, model=model) # type: ignore

def FineTune_MIX22CS2(model='PINN'):
    args = get_args()
    batch_id = ['CS2_35','CS2_36', 'CS2_37', 'CS2_38']
    for batch in batch_id:
        for experiment in range(10):
            setattr(args,'adaptation_lr',0.0007)
            setattr(args,'log_dir','logging.txt')
            setattr(args,'save_folder',f'./results_fine-tuning-v4-model/MIX2-{model} results/{batch}/Experiment{experiment+1}')
            one_adaptation_task(args,source='MIX2',target='CS2',target_batch=batch,model=model) # type: ignore


def FineTune_MIX32MIT(model='PINN'):
    args = get_args()
    
    for experiment in range(10):
        setattr(args,'adaptation_lr',0.0002)
        setattr(args,'log_dir','logging.txt')
        setattr(args,'save_folder',f'./results_fine-tuning/MIX3-MIT-{model}//Experiment{experiment+1}')
        one_adaptation_task(args,source='MIX3',target='MIT',model=model)


def FineTune_MIT2UDDS(model='PINN'):
    args = get_args()
    
    for experiment in range(10):
        setattr(args,'adaptation_lr',0.005)
        setattr(args,'log_dir','logging.txt')
        setattr(args,'save_folder',f'./exp_20260101_Transfer_MIT_UDDS_3/MIT-UDDS-{model}/Experiment{experiment+1}')
        one_adaptation_task(args,source='MIT',target='UDDS',model=model)



def Distill_UDDS(model='PINN'):
    args = get_args()
    
    for experiment in range(10):
        setattr(args,'adaptation_lr',0.02)
        setattr(args,'log_dir','logging.txt')
        setattr(args,'save_folder',f'./exp_20260101_distill_v3/UDDS-{model}/Experiment{experiment+1}')
        one_distill_task(args,source='UDDS',target='UDDS',model=model)



def FineTune():
    args = get_args()
    datasets = ['XJTU', 'TJU', 'HUST', 'MIT']
    batchs = [0, 2, -1, -1]
    for i, source in enumerate(datasets):
        for j, target in enumerate(datasets):
            if source in ['XJTU', 'TJU'] and target in ['XJTU', 'TJU']:
                continue
            if source in ['HUST', 'MIT'] and target in ['HUST', 'MIT']:
                continue
            sb = batchs[i]
            tb = batchs[j]

            for e in range(10):
                setattr(args, 'adaptation_lr', 0.001)
                setattr(args, 'log_dir', f'logging.txt')
                setattr(args, 'save_folder',
                        f'./results_fine-tuning/{source}-{target}/Experiment{e + 1}')
                one_adaptation_task(args, source=source, target=target, source_batch=sb, target_batch=tb)


if __name__ == '__main__':
    # FineTune()
    # FineTune_MIT2HUST()
    # FineTune_HUST2MIT()
    # FineTune_TJU2XJTU()
    # FineTune_XJTU2TJU()

    # FineTune_MIX12NASA('PINN_debug')
    # FineTune_MIX22CS2('PINN_debug')
    # FineTune_MIX32MIT('PINN_v3')

    # FineTune_MIX12NASA('CNN')
    # FineTune_MIX22CS2('CNN')

    # FineTune_MIX12NASA('CNN')
    # FineTune_MIX22CS2('CNN')

    FineTune_MIT2UDDS('Ours')
    # FineTune_MIT2UDDS('MLP')
    # FineTune_MIT2UDDS('CNN')
    # FineTune_MIT2UDDS('LSTM')
    # FineTune_MIT2UDDS('TCN')

    # Distill_UDDS('Ours')

