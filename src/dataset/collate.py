import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Any


def collate_padding(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of (case_bag, control_bag) pairs for processing by zero padding.
    This function pads the bags to the maximum length in the batch and creates masks
    to indicate the real instances and padding.

    Args:
        batch: A list of tuples, where each tuple contains a case bag and a control bag.

    Returns:
        A tuple containing padded case bags, padded control bags, case masks, and control masks.
    """
    case_bags: List[torch.Tensor] = [item[0] for item in batch]  
    control_bags: List[torch.Tensor] = [item[1] for item in batch]

    # Pad bags along the H dimension
    cases_padded_bags: torch.Tensor = pad_sequence(case_bags, batch_first=True, padding_value=0.0)  # Shape [B, max_H, W]
    control_padded_bags: torch.Tensor = pad_sequence(control_bags, batch_first=True, padding_value=0.0)  # Shape [B, max_H, W]

    # Add the singleton channel dim back -> [B, 1, max_H, W]
    cases_padded_bags = cases_padded_bags.unsqueeze(1).to(torch.float32)
    control_padded_bags = control_padded_bags.unsqueeze(1).to(torch.float32)

    # Create mask: 1 for real instances, 0 for padding
    cases_mask: torch.Tensor = torch.tensor(
        [[1] * bag.shape[0] + [0] * (cases_padded_bags.size(2) - bag.shape[0]) for bag in case_bags]
    )

    control_mask: torch.Tensor = torch.tensor(
        [[1] * bag.shape[0] + [0] * (control_padded_bags.size(2) - bag.shape[0]) for bag in control_bags]
    )

    return cases_padded_bags, control_padded_bags, cases_mask, control_mask