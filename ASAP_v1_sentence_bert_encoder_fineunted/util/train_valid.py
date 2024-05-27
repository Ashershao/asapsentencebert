import torch
from util.loss_fn import quality_loss, domain_loss

mse = torch.nn.MSELoss()
cse = torch.nn.CrossEntropyLoss()
dl=quality_loss()
al=domain_loss()

scalar = torch.cuda.amp.GradScaler()

def train_epoch(model,
                device,
                dataloader,
                optimizer,
                batch_size,
                real_batch,
                config='fine-tuned'
                ):
       
    train_loss=0.0
    train_dist_loss=0.0
    train_ang_loss=0.0
    model.train()
    step=0
    update_step=int(real_batch/batch_size)
    for bert_input, nscore,p,score,q in dataloader:
        
        bert_input,nscore,p,q = bert_input.to(device),nscore.to(device),p.to(device),q.to(device)
        with torch.cuda.amp.autocast():    
          if config=='fine-tuned':
              s,emb,_ = model(bert_input)
          else:
              emb,_ =model(bert_input)

        del bert_input 
        
        if (step+1) % update_step==0:

            all_emb=torch.cat((all_emb,emb),dim=0)
            pl=torch.cat((pl,p),dim=0)
            ql=torch.cat((ql,q),dim=0)
            
            with torch.cuda.amp.autocast():            
                
                ang_loss = al(all_emb,pl,0.25,device)                
                dist_loss = dl(all_emb,ql,0.3,device)
                
            del emb,all_emb,pl,ql

            if config =='fine-tuned':
                
                with torch.cuda.amp.autocast():
                    
                    score_loss=mse(s,nscore)
                    loss=score_loss/update_step + 0.4*dist_loss + 0.4*ang_loss

            else:

                with torch.cuda.amp.autocast():
                    
                    loss=dist_loss+ang_loss

            train_loss += loss.item()   
            train_ang_loss += ang_loss.item()
            train_dist_loss += dist_loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        elif (step+1) % update_step==1:
                  
            all_emb=emb
            pl=p
            ql=q

            del emb
              
            if (step+1)== len(dataloader):
                
                with torch.cuda.amp.autocast():    
                
                    ang_loss = al(all_emb,pl,0.25,device)
                    dist_loss = dl(all_emb,ql,0.3,device)
                
                del all_emb,pl,ql

                if config =='fine-tuned':
                
                    with torch.cuda.amp.autocast():    
                        
                        score_loss=mse(s,nscore)
                        loss=score_loss + 0.4*dist_loss + 0.4*ang_loss

                else:

                    with torch.cuda.amp.autocast():    

                        loss=dist_loss+ang_loss

                train_loss += loss.item()   
                train_ang_loss += ang_loss.item()
                train_dist_loss += dist_loss.item()

                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            else:
                
                if config =='fine-tuned':

                    with torch.cuda.amp.autocast():    

                        score_loss=mse(s,nscore)
                        loss=score_loss/update_step

        else:
            

            all_emb=torch.cat((all_emb,emb),dim=0)
            ql=torch.cat((ql,q),dim=0)
            pl=torch.cat((pl,p),dim=0)
                
            if (step+1)== len(dataloader):
                
                with torch.cuda.amp.autocast():    
                    
                    dist_loss = dl(all_emb,ql,0.3,device)
                    ang_loss = al(all_emb,pl,0.25,device)
                
                del all_emb,pl,ql

                if config =='fine-tuned':
            
                    with torch.cuda.amp.autocast():    

                        score_loss=mse(s,nscore)
                        loss=score_loss/update_step + 0.4*dist_loss + 0.4*ang_loss

                else:

                    with torch.cuda.amp.autocast():    


                        loss=dist_loss+ang_loss

                train_loss += loss.item()   
                train_ang_loss += ang_loss.item()
                train_dist_loss += dist_loss.item() 
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            else:
                
                if config =='fine-tuned':
                    
                    with torch.cuda.amp.autocast():    

                        score_loss=mse(s,nscore)
                        loss=score_loss/update_step
        
        step+=1
     
        
    return train_loss ,train_dist_loss , train_ang_loss
    
def valid_epoch(model,
                device,
                dataloader,
                batch_size,
                real_batch,
                config='fine-tuned'
                ):
    val_loss = 0.0
    val_dist_loss= 0.0
    val_ang_loss = 0.0
    model.eval()
    step=0    
    update_step=int(real_batch/batch_size)
    with torch.no_grad():
    
      for bert_input, nscore,p,score,q in dataloader:
            
            bert_input,nscore,p,q = bert_input.to(device),nscore.to(device),p.to(device),q.to(device)

            with torch.cuda.amp.autocast():    
                
                if config=='fine-tuned':
                    s,emb,_ = model(bert_input)
                else:
                    emb,_=model(bert_input)

            del bert_input 
            
            if (step+1) % update_step==0:

                all_emb=torch.cat((all_emb,emb),dim=0)
                pl=torch.cat((pl,p),dim=0)
                ql=torch.cat((ql,q),dim=0)

                with torch.cuda.amp.autocast():    


                    ang_loss = al(all_emb,pl,0.25,device)                
                    dist_loss = dl(all_emb,ql,0.3,device)
                    
                del all_emb,pl,ql

                if config =='fine-tuned':

                    with torch.cuda.amp.autocast():    


                        score_loss=mse(s,nscore)
                        loss=score_loss/update_step + 0.4*dist_loss + 0.4*ang_loss

                else:


                    with torch.cuda.amp.autocast():    

                        loss=dist_loss+ang_loss

                val_loss += loss.item()   
                val_ang_loss += ang_loss.item()
                val_dist_loss += dist_loss.item()
                
            elif (step+1) % update_step==1:
                    
                all_emb=emb
                pl=p
                ql=q

                del emb
                
                if (step+1)== len(dataloader):

                    with torch.cuda.amp.autocast():    
        
                        ang_loss = al(all_emb,pl,0.25,device)
                        dist_loss = dl(all_emb,ql,0.3,device)
                    
                    del all_emb,pl,ql

                    if config =='fine-tuned':

                        with torch.cuda.amp.autocast():    

                            score_loss=mse(s,nscore)
                            loss=score_loss + 0.4*dist_loss + 0.4*ang_loss

                    else:

                        with torch.cuda.amp.autocast():    

                            loss=dist_loss+ang_loss

                    val_loss += loss.item()   
                    val_ang_loss += ang_loss.item()
                    val_dist_loss += dist_loss.item()

                else:
                    if config =='fine-tuned':

                        with torch.cuda.amp.autocast():    

                            score_loss=mse(s,nscore)
                            loss=score_loss/update_step

            else:
                

                all_emb=torch.cat((all_emb,emb),dim=0)
                ql=torch.cat((ql,q),dim=0)
                pl=torch.cat((pl,p),dim=0)
                    
                if (step+1)== len(dataloader):

                    with torch.cuda.amp.autocast():    

                        dist_loss = dl(all_emb,ql,0.3,device)
                        ang_loss = al(all_emb,pl,0.25,device)
                    
                    del emb,all_emb,pl,ql

                    if config =='fine-tuned':

                        with torch.cuda.amp.autocast():    

                            score_loss=mse(s,nscore)
                            loss=score_loss/update_step + 0.4*dist_loss + 0.4*ang_loss

                    else:

                        with torch.cuda.amp.autocast():    
        
                            loss=dist_loss+ang_loss

                    val_loss += loss.item()   
                    val_ang_loss += ang_loss.item()
                    val_dist_loss += dist_loss.item() 
                        
                else:
                    
                    if config =='fine-tuned':

                        with torch.cuda.amp.autocast():    

                            score_loss=mse(s,nscore)
                            loss=score_loss/update_step
            
            step+=1
     
        
    return val_loss ,val_dist_loss , val_ang_loss

    




def predict(model,device,
            dataloader,
            config='fine-tuned'
            ):
    
    model.eval()
    result={1:{'pro_emb':[],'emb':[],'score':[],'pred':[]},
            2:{'pro_emb':[],'emb':[],'score':[],'pred':[]},
            3:{'pro_emb':[],'emb':[],'score':[],'pred':[]},
            4:{'pro_emb':[],'emb':[],'score':[],'pred':[]},
            5:{'pro_emb':[],'emb':[],'score':[],'pred':[]},
            6:{'pro_emb':[],'emb':[],'score':[],'pred':[]},
            7:{'pro_emb':[],'emb':[],'score':[],'pred':[]},
            8:{'pro_emb':[],'emb':[],'score':[],'pred':[]}}
    with torch.no_grad():
        for bert_input, nscore,p,score, q in dataloader:

            bert_input = bert_input.to(device)
            if config=='fine-tuned':
                with torch.cuda.amp.autocast():    

                    s,pro_emb,emb = model(bert_input)
                
                emb=emb.to('cpu').detach().numpy().tolist()
                pro_emb=pro_emb.to('cpu').detach().numpy().tolist()
                s=s.to('cpu').detach().numpy().reshape((len(s)))
                
                for i,prompt in enumerate(p):
                
                    prompt=prompt.item()
                    result[prompt]['pro_emb'].append(pro_emb[i])
                    result[prompt]['score'].append(nscore[i].item())
                    result[prompt]['emb'].append(emb[i])
                    result[prompt]['pred'].append(s[i].item())    
            else:
                
                
                with torch.cuda.amp.autocast():    

                    pro_emb,emb = model(bert_input)
                
                emb=emb.to('cpu').detach().numpy().tolist()
                pro_emb=pro_emb.to('cpu').detach().numpy().tolist()
                
                for i,prompt in enumerate(p):
                
                    prompt=prompt.item()
                    result[prompt]['pro_emb'].append(pro_emb[i])
                    result[prompt]['score'].append(nscore[i].item())
                    result[prompt]['emb'].append(emb[i])

            del p  , nscore , score 
        
    return result