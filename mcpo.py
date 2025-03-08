import torch
import torch.amp as amp
import torch.nn.functional as F
from trl import CPOTrainer


import torch.nn.functional as F

class MCPOTrainer(CPOTrainer):
    
    def cpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the CPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the CPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        logits = (policy_chosen_logps - policy_rejected_logps).to(self.accelerator.device)

        if self.loss_type == "simpo":
            gamma_logratios = self.simpo_gamma / self.beta
            logits = logits - gamma_logratios
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "focal":
            gamma = 2.0  # Focal loss parameter (can be tuned)
            prob = torch.sigmoid(self.beta * logits)  # Convert logits to probabilities
            focal_weight = (1 - prob) ** gamma  # Higher weight for misclassified examples
            losses = (
                -focal_weight * F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - focal_weight * F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        
        elif self.loss_type == "sigmoid_focal":
            gamma = 2.0  # Focal loss parameter
            prob = torch.sigmoid(self.beta * logits)  # Convert logits to probabilities
            focal_weight = (1 - prob) ** gamma  # Emphasize hard-to-classify examples
            sigmoid_loss = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            losses = focal_weight * sigmoid_loss  # Apply focal weighting to sigmoid loss
            
        elif self.loss_type == "focal_bco":
            # Focal loss with KL-divergence-like behavior cloning
            gamma = 2.0  # Focal loss parameter
            prob = torch.sigmoid(self.beta * logits)  # Convert logits to probability
            focal_weight = (1 - prob) ** gamma  # Emphasize hard-to-classify examples

            exp_logits = torch.exp(logits)  # Convert logits to positive values
            kl_loss = -torch.log(exp_logits + 1e-8)  # KL-divergence-like behavior cloning

            losses = focal_weight * kl_loss  # Combine focal weight with behavior cloning
        elif self.loss_type == "sigmoid_bco":
            # Sigmoid loss with KL-divergence-like behavior cloning
            sigmoid_loss = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            exp_logits = torch.exp(logits)  # Convert logits to positive values
            kl_loss = -torch.log(exp_logits + 1e-8)  # KL-divergence-like behavior cloning

            losses = sigmoid_loss * kl_loss  # Combine sigmoid loss with behavior cloning

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'simpo', 'focal', 'sigmoid_focal']"
            )

        chosen_rewards = self.beta * (policy_chosen_logps.to(self.accelerator.device)).detach()
        rejected_rewards = self.beta * (policy_rejected_logps.to(self.accelerator.device)).detach()

        return losses, chosen_rewards, rejected_rewards


