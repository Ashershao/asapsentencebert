
from transformers import AutoModel
import torch

class Encoder_Model(torch.nn.Module):
    def __init__(self,config):
        super(Encoder_Model, self).__init__()
        
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        #self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)             

        self.fcs_ss = torch.nn.Linear(768 , 768)
        self.fcs_s = torch.nn.Linear(768 , 768)
        self.config =config
        
        if config == 'fine-tuned':

            self.fcs_score = torch.nn.Linear(768,1)
            self.sig=torch.nn.Sigmoid()
            
        self.drop=torch.nn.Dropout(0.2)
        
    def forward(self, bert_input):
        #config 0
        #token[batch_size][sequence][3] 0: token 1: type_ids 2: attention mask
        #print(bert_input[:,:,0])
        batch , sentence_len , word_l , e =bert_input.size()[0],bert_input.size()[1],bert_input.size()[2],bert_input.size()[3]       
        bert_input=bert_input.reshape(batch*sentence_len,word_l,e)
        
        bert_output = self.drop(self.bert(input_ids=bert_input[:,:,0],
                                 token_type_ids=bert_input[:,:,1],
                                 attention_mask=bert_input[:,:,2])[0])  
        
        output = torch.mean(bert_output,axis=2)
        output = torch.mean(bert_output,axis=1)
            
        emb = self.fcs_ss(output)
        pro_emb=self.fcs_s(emb)

        if self.config =='fine-tuned':

            s=self.sig(self.fcs_score(emb))

            return s,pro_emb,emb
        
        else:
            return  pro_emb,emb

    

