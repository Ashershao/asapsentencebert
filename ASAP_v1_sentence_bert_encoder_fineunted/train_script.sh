config='fine-tuned pretrained'
for fig in $config;

do
  echo Now run the v1: $fig;
  python Train.py --d=cuda:0 --e=50 --b=1 --rb=64 --mt=$fig --p=10

done
