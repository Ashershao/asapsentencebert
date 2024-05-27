import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from six import string_types
from torch.utils.data import Dataset
from sklearn import manifold
import pandas as pd


score_standard={1:[2,12],2:[1,6],3:[0,3],4:[0,3],5:[0,4],6:[0,4],7:[0,30],8:[0,60]}

def plot_function(train_loss,test_loss,loss_name,total_epoch,path,mt,real_batch,fold):

    range(total_epoch)
    plt.figure(figsize=(10,6))
    plt.plot(range(total_epoch),train_loss,label="Train")
    plt.plot(range(total_epoch),test_loss, color='r',label="Test")
    plt.title("ASAP_quality_doamin_{0}_batch_{1}_fold_{2}_{3}_loss".format(mt,real_batch,fold,loss_name)) # title
    plt.legend()
    plt.ylabel("Loss(mean)") # y label
    plt.xlabel("Epoch")
    plt.savefig(path+'/'+"ASAP_quality_doamin_{0}_batch_{1}_fold_{2}_{3}_loss.jpg".format(mt,real_batch,fold,loss_name))
    #plt.show()

def plot_feature(path,result,feature_name,plot_name,mt,real_batch,fold):
    X=[]
    y=[]
    if plot_name!='score':
        for i in range(1,9):
            for j in result[i][feature_name]:
                X.append(j)
                y.append(i)
    else:
        for i in range(1,9):
            for id,j in enumerate(result[i][feature_name]):
                X.append(j)
                y.append(result[i]['score'][id])
    X=np.array(X)
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

    #Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
    plt.figure(figsize=(16, 16))
    if plot_name == 'score':
        plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap='YlGnBu') 
    else:
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                    fontdict={'weight': 'bold', 'size': 9})
        
    plt.xticks([])
    plt.yticks([])
    plt.title("ASAP_quality_doamin_{0}_batch_{1}_fold_{2}_{3}_{4}_distribution".format(mt,real_batch,fold,feature_name,plot_name)) # title
    plt.savefig(path+'/'+"ASAP_quality_doamin_{0}_batch_{1}_fold_{2}_{3}_{4}_distribution.jpg".format(mt,real_batch,fold,feature_name,plot_name))
    plt.legend()
    #plt.show()


class MyDataset(Dataset):
    def __init__(self,bert_input,score_output,prompt,score,quality):
        self.bert_intput = bert_input
        self.score_output= score_output
        self.prompt=prompt
        self.score=score
        self.quality=quality

    def __getitem__(self,index):
        return self.bert_intput[index],self.score_output[index],self.prompt[index],self.score[index],self.quality[index]

    def __len__(self):
        return len(self.bert_intput)
def return_to_score(prompt,score):
    
    original_score=[]
    for i in score:
        original_score.append(i*(score_standard[prompt][1]-score_standard[prompt][0])+score_standard[prompt][1])
    
    return original_score

def get_average_kappa(result):
    prompt_kappa = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,'avg':0}
    total_kappa = 0 
    for prompt, re in result.items():
        y_t = return_to_score(prompt,re['score'])
        y_p = return_to_score(prompt,re['pred'])
        k= kappa(y_t,y_p,'quadratic')
        prompt_kappa[prompt]=k
        total_kappa += k
    prompt_kappa['avg']=total_kappa/8

    return prompt_kappa
    


def kappa(y_true, y_pred, weights=None, allow_off_by_one=False):
    """
    Calculates the kappa inter-rater agreement between two the gold standard
    and the predicted ratings. Potential values range from -1 (representing
    complete disagreement) to 1 (representing complete agreement).  A kappa
    value of 0 is expected if all agreement is due to chance.
    In the course of calculating kappa, all items in `y_true` and `y_pred` will
    first be converted to floats and then rounded to integers.
    It is assumed that y_true and y_pred contain the complete range of possible
    ratings.
    This function contains a combination of code from yorchopolis's kappa-stats
    and Ben Hamner's Metrics projects on Github.
    :param y_true: The true/actual/gold labels for the data.
    :type y_true: array-like of float
    :param y_pred: The predicted/observed labels for the data.
    :type y_pred: array-like of float
    :param weights: Specifies the weight matrix for the calculation.
                    Options are:
                        -  None = unweighted-kappa
                        -  'quadratic' = quadratic-weighted kappa
                        -  'linear' = linear-weighted kappa
                        -  two-dimensional numpy array = a custom matrix of
                           weights. Each weight corresponds to the
                           :math:`w_{ij}` values in the wikipedia description
                           of how to calculate weighted Cohen's kappa.
    :type weights: str or numpy array
    :param allow_off_by_one: If true, ratings that are off by one are counted as
                             equal, and all other differences are reduced by
                             one. For example, 1 and 2 will be considered to be
                             equal, whereas 1 and 3 will have a difference of 1
                             for when building the weights matrix.
    :type allow_off_by_one: bool
    """
    #logger = logging.getLogger(__name__)

    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    y_true = [int(np.round(float(y))) for y in y_true]
    y_pred = [int(np.round(float(y))) for y in y_pred]


    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    #print(observed)
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    if isinstance(weights, str):
        wt_scheme = weights
        weights = None
    else:
        wt_scheme = ''
    if weights is None:
        weights = np.empty((num_ratings, num_ratings))
        for i in range(num_ratings):
            for j in range(num_ratings):
                diff = abs(i - j)
                if allow_off_by_one and diff:
                    diff -= 1
                if wt_scheme == 'linear':
                    weights[i, j] = diff
                elif wt_scheme == 'quadratic':
                    weights[i, j] = diff ** 2
                elif not wt_scheme:  # unweighted
                    weights[i, j] = bool(diff)
                else:
                    raise ValueError('Invalid weight scheme specified for '
                                     'kappa: {}'.format(wt_scheme))
    #print(weights)

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    #print(hist_true,hist_pred)
    expected = np.outer(hist_true, hist_pred)
    #print(expected)
    # Normalize observed array
    observed = observed / num_scored_items
    #print(observed)
    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    #print(weights*observed)
    #print(weights * expected)
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))
    #print(k)
    return k 

class MyDataset(Dataset):
    def __init__(self,bert_input,score_output,prompt,score,quality):
        self.bert_intput = bert_input
        self.score_output= score_output
        self.prompt=prompt
        self.score=score
        self.quality=quality

    def __getitem__(self,index):
        return self.bert_intput[index],self.score_output[index],self.prompt[index],self.score[index],self.quality[index]

    def __len__(self):
        return len(self.bert_intput)


def output_result(result,path,mt,real_batch):
    prompt_average_qwk={1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    
    for fold,prompt_kappa in enumerate(result):
        for prompt, qwk in prompt_kappa.items():
            if prompt =='avg':
                break
            prompt_average_qwk[prompt]+=qwk
            if fold == 4:
                prompt_average_qwk[prompt]/=5
    

    average_qwk=0
    for qwk in prompt_average_qwk.values():
        average_qwk+=qwk
    average_qwk/=8
    
    prompt_average_qwk['avg']=average_qwk

    print(prompt_average_qwk)

    df=pd.DataFrame.from_dict(prompt_average_qwk,orient='index')

    df.to_excel(path+'/'+"ASAP_quality_doamin_{0}_batch_{1}_result.xlsx".format(mt,real_batch))