import os
import nltk
import pandas as pd
import torch
import numpy as np
import re
from transformers import BertTokenizer

url_replacer = '<url>'
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
score_standard={1:[2,12],2:[1,6],3:[0,3],4:[0,3],5:[0,4],6:[0,4],7:[0,30],8:[0,60]}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def read_data_set(file_path,file_style='excel'):

    if file_style == 'excel':
        data = pd.read_excel(file_path)

    return data
    
def is_number(token):
    return bool(num_regex.match(token))

def replace_url(text):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', url_replacer, text)
    return replaced_text

def tokenize(string):
    tokens = nltk.word_tokenize(string)
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
    return tokens

def text_tokenizer(text, replace_url_flag=True, tokenize_sent_flag=True, create_vocab_flag=False):
    text = replace_url(text)
    text = text.replace(u'"', u'')
    if "..." in text:
        text = re.sub(r'\.{3,}(\s+\.{3,})*', '...', text)        
    if "??" in text:
        text = re.sub(r'\?{2,}(\s+\?{2,})*', '?', text)        
    if "!!" in text:
        text = re.sub(r'\!{2,}(\s+\!{2,})*', '!', text)

    # TODO here
    tokens = tokenize(text)
    if tokenize_sent_flag:
        text = " ".join(tokens)
        sent_tokens = tokenize_to_sentences(text, 50, create_vocab_flag)
        # print sent_tokens
        # sys.exit(0)
        # if not create_vocab_flag:
        #     print "After processed and tokenized, sentence num = %s " % len(sent_tokens)
        return sent_tokens
    else:
        raise NotImplementedError


def tokenize_to_sentences(text, max_sentlength, create_vocab_flag=False):

    # tokenize a long text to a list of sentences
    sents = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', text)

    # Note
    # add special preprocessing for abnormal sentence splitting
    # for example, sentence1 entangled with sentence2 because of period "." connect the end of sentence1 and the begin of sentence2
    # see example: "He is running.He likes the sky". This will be treated as one sentence, needs to be specially processed.
    processed_sents = []
    for sent in sents:
        if re.search(r'(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent):
           
            s = re.split(r'(?=.{2,})(?<=\.{1}|\!|\?|\,)(@?[A-Z]+[a-zA-Z]*[0-9]*)', sent)

            ss = " ".join(s)
   
            ssL = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', ss)

            processed_sents.extend(ssL)
        else:
            processed_sents.append(sent)

    #if create_vocab_flag:
    #    sent_tokens = [tokenize(sent) for sent in processed_sents]
    #    tokens = [w for sent in sent_tokens for w in sent]
        # print tokens
    #    return tokens

    # TODO here
    sent_tokens = []
    for sent in processed_sents:
        shorten_sents_tokens = shorten_sentence(sent, max_sentlength)
        for i in shorten_sents_tokens:
            i=' '.join(i)
            sent_tokens.append(i)
    # if len(sent_tokens) > 90:
    #     print len(sent_tokens), sent_tokens
    return sent_tokens


def shorten_sentence(sent, max_sentlen=50):
    # handling extra long sentence, truncate to no more extra max_sentlen
    new_tokens = []
    sent = sent.strip()
    tokens = nltk.word_tokenize(sent)
    if len(tokens) > max_sentlen:
        # print len(tokens)
        # Step 1: split sentence based on keywords
        # split_keywords = ['because', 'but', 'so', 'then', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        split_keywords = ['because', 'but', 'so', 'You', 'He', 'She', 'We', 'It', 'They', 'Your', 'His', 'Her']
        k_indexes = [i for i, key in enumerate(tokens) if key in split_keywords]
        processed_tokens = []
        if not k_indexes:
            num = int(len(tokens) / max_sentlen)
            k_indexes = [(i+1)*max_sentlen for i in range(num)]

        processed_tokens.append(tokens[0:k_indexes[0]])
        len_k = len(k_indexes)
        for j in range(len_k-1):
            processed_tokens.append(tokens[k_indexes[j]:k_indexes[j+1]])
        processed_tokens.append(tokens[k_indexes[-1]:])

        # Step 2: split sentence to no more than max_sentlen
        # if there are still sentences whose length exceeds max_sentlen
        for token in processed_tokens:
            if len(token) > max_sentlen:
                num = int(len(token) / max_sentlen)
                s_indexes = [(i+1)*max_sentlen for i in range(num)]

                len_s = len(s_indexes)
                new_tokens.append(token[0:s_indexes[0]])
                for j in range(len_s-1):
                    new_tokens.append(token[s_indexes[j]:s_indexes[j+1]])
                new_tokens.append(token[s_indexes[-1]:])

            else:
                new_tokens.append(token)
    else:
            return [tokens]

    # print "Before processed sentences length = %d, after processed sentences num = %d " % (len(tokens), len(new_tokens))
    return new_tokens

def read_file(path,data):
    fold_dataset_id={}
    for i in range(5): 
        data_dict={}
        id_list=[]
        f = open(path+str(i)+'/train_ids.txt')
        for line in f.readlines():
            ids=int(line.replace('\n', '').replace('\r', ''))
            if ids in list(data['essay_id']):
                id_list.append(ids)
        f.close
        data_dict['train']=id_list

        id_list=[]
        f = open(path+str(i)+'/dev_ids.txt')
        for line in f.readlines():
            ids=int(line.replace('\n', '').replace('\r', ''))
            if ids in list(data['essay_id']):
                id_list.append(ids)
        f.close
        data_dict['dev']=id_list

        id_list=[]
        f = open(path+str(i)+'/test_ids.txt')
        for line in f.readlines():
            ids=int(line.replace('\n', '').replace('\r', ''))
            if ids in list(data['essay_id']):
                id_list.append(ids)
        f.close
        data_dict['test']=id_list

        fold_dataset_id[i]=data_dict
    
    return fold_dataset_id

def preprocessing_as_input(text_list):
    
    essay_list=[]
    
    for i in text_list:
        sent=text_tokenizer(i)
        essay_list.append(sent)

    
    bert_inputs=[]
    data_type_ids=[]
    data_attention_mask=[]
    for essay in essay_list: 
        
        sent_inputs=[]
        sent_data_type_ids=[]
        sent_data_attention_mask=[]

        for i in range(97):
            if i < len(essay):
                inputs = tokenizer(essay[i], padding = 'max_length', 
                                    max_length = 512, return_tensors="pt", truncation = True)
                sent_inputs.extend(inputs['input_ids'].numpy().tolist())
                sent_data_type_ids.extend(inputs['token_type_ids'].numpy().tolist())
                sent_data_attention_mask.extend(inputs['attention_mask'].numpy().tolist())
            else:
                inputs = tokenizer('', padding = 'max_length', 
                                    max_length = 512, return_tensors="pt", truncation = True)
                sent_inputs.extend(inputs['input_ids'].numpy().tolist())
                sent_data_type_ids.extend(inputs['token_type_ids'].numpy().tolist())
                sent_data_attention_mask.extend(inputs['attention_mask'].numpy().tolist())

        bert_inputs.append(sent_inputs)
        data_type_ids.append(sent_data_type_ids)
        data_attention_mask.append(sent_data_attention_mask)
        
    return  bert_inputs, data_type_ids, data_attention_mask

def get_tensor_input (data_input,data_type_ids,data_attention_mask,score):

    bert_input = torch.zeros((len(data_input),len(data_input[0]),512,3),dtype=torch.long,device='cpu') # torch embedding type need to long or int
    bert_input[:,:,:,0]=torch.tensor(data_input)
    bert_input[:,:,:,1]=torch.tensor(data_type_ids)
    bert_input[:,:,:,2]=torch.tensor(data_attention_mask)
    score_output= torch.tensor(score)
    score_output= torch.unsqueeze(score_output, 1)

    return bert_input,score_output

def data_prepare(path,file_path,file_style,interval):

    interval = 1 / interval
    data=read_data_set(file_path,file_style='excel')    
    data.dropna(axis=0,subset=['domain1_score'],inplace=True)
        
    fold_dataset_id=read_file(path,data)
    data.index=data['essay_id']


    prompt_fold={0:{},1:{},2:{},3:{},4:{}}
    for i in range(5):
        #print(i)
        data_dict={'train':{},'dev':{},'test':{}}
            
        train_id=list(data['essay_id'].loc[fold_dataset_id[i]['train']])
        train_text=list(data['essay'].loc[fold_dataset_id[i]['train']])
        train_prompt=list(data['essay_set'].loc[fold_dataset_id[i]['train']])
        train_score=list(data['domain1_score'].loc[fold_dataset_id[i]['train']])
        train_score_n=[(j-score_standard[train_prompt[ids]][0])/(score_standard[train_prompt[ids]][1]-score_standard[train_prompt[ids]][0]) for ids,j in enumerate(train_score)]
        train_score_quality=[ 0 if i>=0.8 else 1 if i<=0.3 else 2 for i in train_score_n]
        bert_inputs, data_type_ids, data_attention_mask = preprocessing_as_input(train_text)
        bert_input, train_score_n = get_tensor_input (bert_inputs,data_type_ids,data_attention_mask,train_score_n)

        data_dict['train']['id']=train_id
        data_dict['train']['prompt']=train_prompt
        data_dict['train']['score']=train_score
        data_dict['train']['normalize score']=train_score_n
        data_dict['train']['text code']=bert_input
        data_dict['train']['quality']=train_score_quality
        
        dev_id=list(data['essay_id'].loc[fold_dataset_id[i]['dev']])
        dev_text=list(data['essay'].loc[fold_dataset_id[i]['dev']])
        dev_prompt=list(data['essay_set'].loc[fold_dataset_id[i]['dev']])
        dev_score=list(data['domain1_score'].loc[fold_dataset_id[i]['dev']])
        dev_score_n=[(j-score_standard[dev_prompt[ids]][0])/(score_standard[dev_prompt[ids]][1]-score_standard[dev_prompt[ids]][0]) for ids,j in enumerate(dev_score)]
        dev_score_quality=[  0 if i>=0.8 else 1 if i<=0.3 else 2 for i in dev_score_n]
        bert_inputs, data_type_ids, data_attention_mask = preprocessing_as_input(dev_text)
        bert_input, dev_score_n = get_tensor_input (bert_inputs,data_type_ids,data_attention_mask,dev_score_n)
        
        data_dict['dev']['id']=dev_id
        data_dict['dev']['prompt']=dev_prompt
        data_dict['dev']['score']=dev_score
        data_dict['dev']['normalize score']=dev_score_n
        data_dict['dev']['text code']=bert_input
        data_dict['dev']['quality']=dev_score_quality
        
        test_id=list(data['essay_id'].loc[fold_dataset_id[i]['test']])
        test_text=list(data['essay'].loc[fold_dataset_id[i]['test']])
        test_prompt=list(data['essay_set'].loc[fold_dataset_id[i]['test']])
        test_score=list(data['domain1_score'].loc[fold_dataset_id[i]['test']])
        test_score_n=[(j-score_standard[test_prompt[ids]][0])/(score_standard[test_prompt[ids]][1]-score_standard[test_prompt[ids]][0]) for ids,j in enumerate(test_score)]
        test_score_quality=[  0 if i>=0.8 else 1 if i<=0.3 else 2 for i in test_score_n]
        bert_inputs, data_type_ids, data_attention_mask = preprocessing_as_input(test_text)
        bert_input, test_score_n = get_tensor_input (bert_inputs,data_type_ids,data_attention_mask,test_score_n)
        
        data_dict['test']['id']=test_id
        data_dict['test']['prompt']=test_prompt
        data_dict['test']['score']=test_score
        data_dict['test']['normalize score']=test_score_n
        data_dict['test']['text code']=bert_input
        data_dict['test']['quality']=test_score_quality
        
        prompt_fold[i]=data_dict

    return prompt_fold

