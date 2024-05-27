import torch


def pairwise_distance_torch(embeddings,device):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)
    
    c1 = torch.sqrt(torch.pow(precise_embeddings, 2).sum(axis=-1))
    c2 = torch.sqrt(torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0))
    c1 = c1.reshape((c1.shape[0], 1))
    c2 = c2.reshape((1, c2.shape[0]))
    # length ||v1|-|v2||
    pairwise_distances = torch.abs(c1 - c2)    

    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones((pairwise_distances.shape[0], pairwise_distances.shape[1])) - torch.diag(torch.ones(pairwise_distances.shape[0]))
    pairwise_distances = torch.mul(pairwise_distances.to(device), mask_offdiagonals.to(device))
    
    del precise_embeddings,c1,c2,mask_offdiagonals
    
    return pairwise_distances

def pairwise_cosine_torch(embeddings):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      embeddings: 2-D Tensor of size [number of data, feature dimension].
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """

    # pairwise distance matrix with precise embeddings
    precise_embeddings = embeddings.to(dtype=torch.float32)
    
    c1 = torch.pow(precise_embeddings, 2).sum(axis=-1)
    c2 = torch.pow(precise_embeddings.transpose(0, 1), 2).sum(axis=0)
    c3 = precise_embeddings @ precise_embeddings.transpose(0, 1)
    
    c1 = torch.sqrt(c1.reshape((c1.shape[0], 1)))
    #print(c1)
    c2 = torch.sqrt(c2.reshape((1, c2.shape[0])))
    #print(c2)
    c12 = torch.matmul(c1,c2)
    #print(c12)
    pairwise_angle = c3/c12
    #print(pairwise_angle)
    
    del precise_embeddings,c1,c2,c3,c12
    
    return pairwise_angle

def weight_calculation(pair_loss,sig=1,mu=3):
    pair_max=torch.max(pair_loss,1)[0].unsqueeze(-1)
    #print(pair_max)
    weight=0.5*(1+torch.special.erf((pair_loss-mu)/(sig*2**0.5)))
    #print('weight:',weight)
    
    return weight

class quality_loss(torch.nn.Module):
    def __init__(self):
        super(quality_loss, self).__init__()

    def forward(self, represent, q, m,d):
        
        pdist_matrix=pairwise_distance_torch(represent,d)
        #print('norm diff matrix:',pdist_matrix)
        q=q.view(q.shape[0],1)
        #print(q.shape)
        q_count = torch.where(q!=2,1,0)
        q_mask = torch.eq(q_count,q_count.transpose(0,1))
        pos_mask= torch.eq(q, q.transpose(0, 1))
        neg_mask= torch.eq(q, q.transpose(0, 1)).logical_not()
        
        #print('pos mask:',pos_mask)
        #print('neg mask:',neg_mask)
        
        zero=torch.zeros(pdist_matrix.shape)
        zero=zero.to(d)
        
        pos_pair_loss = torch.mul(pos_mask,pdist_matrix)
        neg_pair_loss = torch.maximum(zero,torch.mul(neg_mask,m-pdist_matrix))
        
        #print('pos pair loss:',pos_pair_loss)
        #print('neg pair loss',neg_pair_loss)
        
        pos_weight=weight_calculation(pos_pair_loss,0.5,0.2)
        neg_weight=weight_calculation(neg_pair_loss,0.05,0.05)
        
        
        #print('pos weight:',pos_weight)
        #print('neg weight',neg_weight)
        
        #print(torch.mul(pos_weight,pos_pair_loss))
        #print(torch.mul(neg_weight,neg_pair_loss)) 
        loss_matrix= torch.mul(pos_weight,pos_pair_loss)+torch.mul(neg_weight,neg_pair_loss)
        loss_matrix=torch.mul(loss_matrix,q_mask)
        #loss_matrix= pos_pair_loss+neg_pair_loss
        
        if q.shape[0]==1:
            loss = torch.zeros(1).to(d)
        else:    
            loss=loss_matrix.sum()/(torch.numel(loss_matrix)-represent.shape[0])
        #print(loss)
        
        del pdist_matrix,neg_mask,pos_mask,pos_pair_loss,neg_pair_loss,loss_matrix#,pos_weight,neg_weight
        
        return loss

class domain_loss(torch.nn.Module):
    def __init__(self):
        super(domain_loss, self).__init__()

    def forward(self, represent, p,m,d):
        
        pangle_matrix=pairwise_cosine_torch(represent)
        # Build pairwise binary adjacency matrix.
        p=p.view(p.shape[0],1)
        pos_mask= torch.eq(p, p.transpose(0, 1))
        neg_mask= torch.eq(p, p.transpose(0, 1)).logical_not()
        
        zero=torch.zeros(pangle_matrix.shape)
        zero=zero.to(d)
        
        pos_pair_loss = 1-torch.mul(pos_mask,pangle_matrix)
        neg_pair_loss = torch.maximum(zero,torch.mul(neg_mask,pangle_matrix)-m)
        
        pos_weight=weight_calculation(pos_pair_loss,0.5,0.2)
        neg_weight=weight_calculation(neg_pair_loss,0.2,0.2)
                 
        loss_matrix= torch.mul(pos_weight,pos_pair_loss)+torch.mul(neg_weight,neg_pair_loss)
        #loss_matrix= pos_pair_loss+neg_pair_loss
        
        if p.shape[0]==1:
            loss = torch.zeros(1).to(d)
        else: 
            loss=loss_matrix.sum()/(torch.numel(loss_matrix)-represent.shape[0])
        
        del pangle_matrix,pos_mask,neg_mask,pos_pair_loss,neg_pair_loss,loss_matrix#,pos_weight,neg_weight
        #print(loss)
        return loss

class decouple_contrastive_loss(torch.nn.Module):
    def __init__(self):
        super(decouple_contrastive_loss, self).__init__()

    def forward(self, represent, label,t,d):
        
        pangle_matrix=pairwise_cosine_torch(represent)
        # Build pairwise binary adjacency matrix.
        label=label.view(label.shape[0],1)
        pos_mask= torch.eq(label, label.transpose(0, 1))
        neg_mask= torch.eq(label, label.transpose(0, 1)).logical_not()
        
        zero=torch.zeros(pangle_matrix.shape)
        zero=zero.to(d)
        
        pos_pair_loss = torch.exp(torch.mul(pos_mask,pangle_matrix/t)).sum(axis=-1)
        neg_pair_loss = torch.exp(torch.mul(neg_mask,pangle_matrix/t)).sum(axis=-1)
        

         
        loss=torch.log(neg_pair_loss)-torch.log(pos_pair_loss)

        loss=loss.sum()/represent.shape[0]
        
        del pangle_matrix,pos_mask,neg_mask,pos_pair_loss,neg_pair_loss#,pos_weight,neg_weight
        #print(loss)
        return loss