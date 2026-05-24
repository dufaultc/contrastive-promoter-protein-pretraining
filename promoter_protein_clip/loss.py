import torch
import torch.nn as nn


class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=0.07, device=None):
        super().__init__()        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / logit_scale)))

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.to(self.device)

    def forward(self, protein_emb, promoter_emb):
        logit_scale = self.logit_scale.exp()

        logits_per_protein = logit_scale * protein_emb @ promoter_emb.t()
        logits_per_promoter = logits_per_protein.t()
        batch_size = logits_per_protein.shape[0]

        # Create the ground-truth labels. The (i, i) element is the correct pair.
        labels = torch.arange(batch_size, dtype=torch.long, device=logits_per_protein.device)
        
        #print("labels", labels)

        # Cross-entropy loss for protein->promoter and promoter->protein
        loss_fn = nn.CrossEntropyLoss()
        loss_p = loss_fn(logits_per_protein, labels)
        loss_m = loss_fn(logits_per_promoter, labels)
        loss = (loss_p + loss_m) / 2.0            
        return loss