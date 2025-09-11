import torch
import torch.nn as nn
import itertools

class AllPairsHingeLoss(nn.Module):
    """
    Calculates the batch-wise hinge loss for all unique pairs of embeddings.
    Supports hard negative mining by percentage.
    """
    def __init__(self, alpha=0.2, neg_sample_percent=None):
        """
        Initializes the loss module.
        Args:
            alpha (float): The margin for the hinge loss.
            neg_sample_percent (float, optional): Percentage of hardest negative samples to use (0.0 to 1.0).
                                                 If None or 1.0, all negatives are used.
        """
        super(AllPairsHingeLoss, self).__init__()
        self.alpha = alpha
        self.neg_sample_percent = neg_sample_percent

    def forward(self, embeddings_dict):
        """
        Args:
            embeddings_dict (dict): A dictionary where keys are modality names (str)
                                     and values are the corresponding embeddings
                                     (torch.Tensor of shape [batch_size, embedding_dim]).
                                     Embeddings should be L2-normalized.

        Returns:
            torch.Tensor: The total hinge loss, averaged over all pairs and the batch.
        """
        total_loss = 0
        num_pairs = 0
        
        # Get all unique pairs of modalities
        modality_keys = list(embeddings_dict.keys())
        for key1, key2 in itertools.combinations(modality_keys, 2):
            embeddings_1 = embeddings_dict[key1]
            embeddings_2 = embeddings_dict[key2]
            
            # Add to the total loss
            total_loss += self.batch_hinge_loss(embeddings_1, embeddings_2)
            num_pairs += 1
            
        # Average the loss over the number of pairs
        if num_pairs == 0:
            return torch.tensor(0.0, device=next(iter(embeddings_dict.values())).device)
        
        return total_loss / num_pairs

    def batch_hinge_loss(self, embeddings_1, embeddings_2):
        """
        Calculates the symmetric hinge loss for a single pair of embedding tensors.
        """
        batch_size = embeddings_1.size(0)
        
        # Calculate cosine similarity. Assumes embeddings are L2 normalized.
        # Shape: [batch_size, batch_size]
        similarity_matrix = torch.matmul(embeddings_1, embeddings_2.t())

        # Get the similarity of the correct (diagonal) pairs
        # Shape: [batch_size]
        diag = similarity_matrix.diag()

        # --- Calculate loss for (embeddings_1 -> embeddings_2) ---
        cost_1 = torch.clamp(self.alpha - similarity_matrix + diag.view(-1, 1), min=0)
        
        # --- Calculate loss for (embeddings_2 -> embeddings_1) ---
        cost_2 = torch.clamp(self.alpha - similarity_matrix.t() + diag.view(-1, 1), min=0)

        # Zero out the loss for correct pairs
        I = torch.eye(batch_size, device=embeddings_1.device).bool()
        cost_1 = cost_1.masked_fill(I, 0)
        cost_2 = cost_2.masked_fill(I, 0)

        # Hard negative mining by percentage
        if self.neg_sample_percent is not None and 0.0 < self.neg_sample_percent < 1.0:
            # Number of negative samples per item is (batch_size - 1)
            num_negatives = batch_size - 1
            k = int(num_negatives * self.neg_sample_percent)
            # Ensure at least one sample is taken if percentage > 0
            k = max(1, k)
            
            # Sort losses and take the top k for each item
            cost_1 = cost_1.sort(dim=1, descending=True)[0][:, :k]
            cost_2 = cost_2.sort(dim=1, descending=True)[0][:, :k]

        # Sum the loss over all incorrect pairs and average by batch size
        total_pair_loss = (cost_1.sum() + cost_2.sum()) / batch_size
        
        return total_pair_loss
