How to run the code:

python Train.py" --d=cuda:0 --e=30 --b=4 --rb=32 --m=Bert --a=bottlencek --do=True --qu=True --lt=qldl

The detail argument statement:
-h, --help  show this help message and exit
  --d D       device| Determine to use gpu or cpu
  --e E       epoch| total epoch you want to run 
  --b B       batch| batch size give to computer 
  --rb RB     real batch| batch size for calculate for update loss function 
  --m M       Bert or LSTM | Enocder  Bert: only bert  LSTM: add Bi-LSTM after Bert
  --a A       adapter: bottleneck/lora/freeze | decide adapter you want to run. Note: the freeze only fixed bert parameter.
  --do DO     domain activate or not | need domain 
  --qu QU     quality activate or not | need quality
  --lt LT     loss type: qldl/decouple or None | None only consider MSE

  Note about domain and quality:
  1. if domain and quality are False. The model only train bert and add linear for MSE.
  2. if only domain = True. The model only consider domain loss
  3. if only quality = True. The model only consider quality True
  4. if both are Ture. The model is shared the same bert parameters and add two different linear for quality and domain 