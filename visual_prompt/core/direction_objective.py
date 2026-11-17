"""Optimize one support-derived direction; encoder parameters stay frozen."""
import torch
import torch.nn.functional as F


def direction_loss(d, target, background, high_freq, bright, cluster_weight,
                   hard_weight, threshold=0.1):
    cluster = (d @ background.T).max()
    hf = (d @ high_freq.T).max() if len(high_freq) else d.sum()*0
    bv = (d @ bright.T).max() if len(bright) else d.sum()*0
    loss = -(d*target).sum() + cluster_weight*cluster
    loss = loss + hard_weight*(F.relu(hf-threshold)+F.relu(bv-threshold))
    return loss


def optimize_direction(target, background, high_freq, bright, cluster_weight=0.05,
                       hard_weight=0.6, threshold=0.1, steps=300, learning_rate=0.02):
    target=F.normalize(target.detach().float().reshape(1,-1),dim=1)
    groups=[F.normalize(g.detach().float(),dim=1) for g in (background,high_freq,bright)]
    with torch.enable_grad():
        d=torch.nn.Parameter(target.clone())
        optimizer=torch.optim.Adam([d],lr=learning_rate)
        best=target.clone();best_loss=float('inf');history=[]
        for step in range(steps+1):
            loss=direction_loss(d,target,*groups,cluster_weight,hard_weight,threshold)
            value=float(loss.detach().item())
            if value<best_loss:
                best_loss=value;best=d.detach().clone()
            if step%50==0 or step==steps:history.append({'step':step,'loss':value})
            if step==steps:break
            optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():d.copy_(F.normalize(d,dim=1))
    with torch.no_grad():
        return best, {'cluster_weight':cluster_weight,'hard_weight':hard_weight,'threshold':threshold,
                      'steps':steps,'best_loss':best_loss,'target_cosine':float((best*target).sum().item()),
                      'max_background_cosine':float((best@groups[0].T).max().item()),
                      'max_high_freq_cosine':float((best@groups[1].T).max().item()) if len(groups[1]) else None,
                      'max_bright_cosine':float((best@groups[2].T).max().item()) if len(groups[2]) else None,
                      'history':history}
