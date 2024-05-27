from util.model import Encoder_Model
from util import preprocessing
from util import train_valid, utils
import torch
from transformers import AdamW
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--d',type=str, required=True, default="cuda:0",help='device')
parser.add_argument('--e',type=int, required=True, default=30,help='epoch')
parser.add_argument('--b',type=int, required=True, default=16,help='batch')
parser.add_argument('--rb',type=int, required=True, default=64,help='real batch')
parser.add_argument('--mt',type=str, required=True, default='pre_trained',help='pre_train or fine-tuned')
parser.add_argument('--p',type=int, required=False,default=5,help='how many type of quality label')
args = parser.parse_args()
print(args)


def main():

    device = torch.device( args.d if torch.cuda.is_available() else "cpu")
    #config
    total_epoch = args.e
    batch_size = args.b
    real_batch = args.rb
    mt = args.mt
    partition=args.p

    root = 'ASAP_quality_doamin_{0}_BERT_batch_{1}_label_origin'.format(mt,real_batch)
    data=preprocessing.data_prepare('fold_','asap-aes/training_set_rel3.xlsx','excel',partition)
    
    performance_qwk=[]
    prompt_path = root 
    if not os.path.exists(prompt_path):
        os.makedirs(prompt_path)
    for fold,detail in data.items():

        x_train,y_train,p_train,s_train,q_train=detail['train']['text code'],detail['train']['normalize score'],detail['train']['prompt'],detail['train']['score'],detail['train']['quality']
        x_test,y_test,p_test,s_test,q_test=detail['test']['text code'],detail['test']['normalize score'],detail['test']['prompt'],detail['test']['score'],detail['test']['quality']
        x_dev,y_dev,p_dev,s_dev,q_dev=detail['dev']['text code'],detail['dev']['normalize score'],detail['dev']['prompt'],detail['dev']['score'],detail['dev']['quality']

        train_dataset = utils.MyDataset(x_train,y_train,p_train,s_train,q_train)
        test_dataset = utils.MyDataset(x_test,y_test,p_test,s_test,q_test)
        dev_dataset = utils.MyDataset(x_dev,y_dev,p_dev,s_dev,q_dev)

        train_sampler = SubsetRandomSampler(range(len(x_train)))
        test_sampler = SubsetRandomSampler(range(len(x_test)))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
        
        history = {'train_loss': [],'train_dist_loss': [],'train_ang_loss': [],
                   'test_loss': [],'test_dist_loss': [],'test_ang_loss': []}

        mod=Encoder_Model(config=mt)
        mod.to(device).half()
        print("model torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("model input torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("model input torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print('_____________________________________________________________')
        params= mod.named_parameters()
        no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
        grouped_params = [
                {
                    'params' : [p for n , p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay':0.01,
                    'lr':0.00001,
                    'ori_lr': 0.00001
                },
                {
                    'params' : [p for n , p in params if any(nd in n for nd in no_decay)],
                    'weight_decay':0.0,
                    'lr':0.00001,
                    'ori_lr': 0.00001
                }
            ]            

        optimizer = optim.AdamW(grouped_params) 

        #scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        best_test=1000
        best_kappa=0
        for epoch in range(total_epoch):
            
            train_result=train_valid.train_epoch(model = mod,
                                                device=device,
                                                dataloader = train_loader,
                                                optimizer=optimizer,
                                                batch_size=batch_size,
                                                real_batch=real_batch,
                                                config=mt)
            test_result=train_valid.valid_epoch(model = mod,
                                                device=device,
                                                dataloader = test_loader,
                                                batch_size=batch_size,
                                                real_batch=real_batch,
                                                config=mt)


            train_loss = train_result[0] / len(train_loader.sampler)
            train_dist_loss = train_result[1] / len(train_loader.sampler)
            train_ang_loss = train_result[2] // len(train_loader.sampler)
            test_loss = test_result[0] / len(test_loader.sampler)
            test_dist_loss = test_result[1] / len(test_loader.sampler)
            test_ang_loss = test_result[2] // len(test_loader.sampler)
                
            history['train_loss'].append(train_loss)
            history['train_dist_loss'].append(train_dist_loss)
            history['train_ang_loss'].append(train_ang_loss)
            history['test_loss'].append(test_loss)
            history['test_dist_loss'].append(test_dist_loss)
            history['test_ang_loss'].append(test_ang_loss)



            print("Epoch:{}/{} AVG Training Loss:{:.6f} AVG Test Loss:{:.6f} %".format(epoch + 1,
                                                                                            total_epoch,
                                                                                            train_loss,
                                                                                            test_loss))  
            if mt == 'fine-tuned':

                train_re=train_valid.predict(model=mod,
                                    device=device,
                                    dataloader = train_loader,
                                    config=mt)
                train_kappa=utils.get_average_kappa(train_re)
                    
                test_re=train_valid.predict(model=mod,
                                    device=device,
                                    dataloader = test_loader,
                                    config=mt)
                test_kappa=utils.get_average_kappa(test_re)

                if test_kappa['avg']>best_kappa:
                    print('test kappa {0}'.format(test_kappa['avg']))
                    dev_result=train_valid.valid_epoch(model = mod,
                                                    device=device,
                                                    dataloader = dev_loader,
                                                    batch_size=batch_size,
                                                    real_batch=real_batch,
                                                    config=mt)
                    print('_____________________________________________________________')
                    print('dev lost:{0}'.format(dev_result[0]/len(dev_loader.sampler)*real_batch/batch_size))
                    print('_____________________________________________________________')

                    torch.save(mod.state_dict(),'{0}/ASAP_quality_doamin_{1}_BERT_batch_{2}_label_origin_fold_{3}'.format(prompt_path,mt,real_batch,fold))
                    result=train_valid.predict(model = mod,
                                                device=device,
                                                dataloader = dev_loader,
                                                config=mt
                                                )
                    result_kappa=utils.get_average_kappa(result)
                    print('_______________________________________________________________')
                    print('dev kappa {0}'.format(result_kappa['avg']))
                    print('_______________________________________________________________')
                    best_kappa=test_kappa['avg']                   
                
                
            else:

                if test_loss < best_test:

                    dev_result=train_valid.valid_epoch(model = mod,
                                                    device=device,
                                                    dataloader = dev_loader,
                                                    batch_size=batch_size,
                                                    real_batch=real_batch,
                                                    config=mt)
                    print('_____________________________________________________________')
                    print('dev lost:{0}'.format(dev_result[0]/len(dev_loader.sampler)*real_batch/batch_size))
                    print('_____________________________________________________________')

                    torch.save(mod.state_dict(),'{0}/ASAP_quality_doamin_{1}_BERT_batch_{2}_label_origin_fold_{3}'.format(prompt_path,mt,real_batch,fold))
                    result=train_valid.predict(model = mod,
                                                device=device,
                                                dataloader = dev_loader,
                                                config=mt
                                                )
                    best_test=test_loss
                    
            
            #scheduler.step()
        if mt == 'fine-tuned':
            performance_qwk.append(result_kappa)
        
        utils.plot_function(history['train_loss'],history['test_loss'],'total_loss',total_epoch,prompt_path,mt,real_batch,fold)
        utils.plot_function(history['train_dist_loss'],history['test_dist_loss'],'dist_loss',total_epoch,prompt_path,mt,real_batch,fold)
        utils.plot_function(history['train_ang_loss'],history['test_ang_loss'],'ang_loss',total_epoch,prompt_path,mt,real_batch,fold) 
        utils.plot_feature(prompt_path,result,'emb','score',mt,real_batch,fold) 
        utils.plot_feature(prompt_path,result,'emb','prompt',mt,real_batch,fold)
        utils.plot_feature(prompt_path,result,'pro_emb','score',mt,real_batch,fold) 
        utils.plot_feature(prompt_path,result,'pro_emb','prompt',mt,real_batch,fold)

    if mt == 'fine-tune':

        utils.output_result(performance_qwk,prompt_path,mt,real_batch)


if __name__ == '__main__':

    main()
