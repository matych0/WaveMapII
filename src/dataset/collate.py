import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict


def collate_padding(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of (case_bag, control_bag) pairs for processing by zero padding.
    This function pads the bags to the maximum length in the batch and creates masks
    to indicate the real instances and padding.

    Args:
        batch: A list of tuples, where each tuple contains a case bag and a control bag.

    Returns:
        A tuple containing padded case bags, padded control bags, case masks, and control masks.
    """
    traces_bags: List[torch.Tensor] = [item[0] for item in batch]
    durations: torch.Tensor = torch.stack([item[1] for item in batch])
    events: torch.Tensor = torch.stack([item[2] for item in batch])

    # Pad bags along the H dimension
    traces_padded_bags: torch.Tensor = pad_sequence(traces_bags, batch_first=True, padding_value=0.0)  # Shape [B, max_H, W]

    # Add the singleton channel dim back -> [B, 1, max_H, W]
    traces_padded_bags = traces_padded_bags.unsqueeze(1).to(torch.float32)

    # Create mask: 1 for real instances, 0 for padding
    traces_mask: torch.Tensor = torch.tensor(
        [[1] * bag.shape[0] + [0] * (traces_padded_bags.size(2) - bag.shape[0]) for bag in traces_bags]
    )

    return traces_padded_bags, traces_mask, durations, events

def collate_padding_merged(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    case_bags.extend(control_bags)

    # Pad bags along the H dimension
    cases_padded_bags: torch.Tensor = pad_sequence(case_bags, batch_first=True, padding_value=0.0)  # Shape [B, max_H, W]

    # Add the singleton channel dim back -> [B, 1, max_H, W]
    cases_padded_bags = cases_padded_bags.unsqueeze(1).to(torch.float32)

    # Create mask: 1 for real instances, 0 for padding
    cases_mask: torch.Tensor = torch.tensor(
        [[1] * bag.shape[0] + [0] * (cases_padded_bags.size(2) - bag.shape[0]) for bag in case_bags]
    )

    return cases_padded_bags, cases_mask


def collate_padding_two_tensors(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def collate_validation_inference(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of (case_bag, control_bag, mask) tuples for processing by zero padding.
    This function pads the bags to the maximum length in the batch and creates masks
    to indicate the real instances and padding.

    Args:
        batch: A list of tuples, where each tuple contains a case bag, control bag, and mask.

    Returns:
        A tuple containing padded case bags, padded control bags, case masks, and control masks.
    """
    durations: List[torch.Tensor] = [item[0] for item in batch]  # Extract events from the batch
    events: List[torch.Tensor] = [item[1] for item in batch]  # Extract durations from the batch
    traces_bags: List[torch.Tensor] = [item[2] for item in batch]  # Extract trace bags from the batch
    traces_orig: List[torch.Tensor] = [item[3] for item in batch]  # Extract original traces from the batch
    filepaths: List[str] = [item[4] for item in batch]  # Extract file paths from the batch
    trace_indices: List[int] = [item[5] for item in batch]  # Extract trace indices from the batch
    ptp: List[torch.Tensor] = [item[6] for item in batch]  # Extract peak-to-peak values from the batch

    # Pad bags along the H dimension
    traces_padded_bags: torch.Tensor = pad_sequence(traces_bags, batch_first=True, padding_value=0.0)  # Shape [B, max_H, W]
    traces_orig_padded_bags: torch.Tensor = pad_sequence(traces_orig, batch_first=True, padding_value=0.0)  # Shape [B, max_H, W]

    # Add the singleton channel dim back -> [B, 1, max_H, W]
    traces_padded_bags = traces_padded_bags.unsqueeze(1).to(torch.float32)
    traces_orig_padded_bags = traces_orig_padded_bags.unsqueeze(1).to(torch.float32)

    # Create mask: 1 for real instances, 0 for padding
    traces_masks: torch.Tensor = torch.tensor(
        [[1] * bag.shape[0] + [0] * (traces_padded_bags.size(2) - bag.shape[0]) for bag in traces_bags]
    )

    return durations, events, traces_padded_bags, traces_orig_padded_bags, traces_masks, filepaths, trace_indices, ptp


def collate_validation(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collates a batch of (case_bag, control_bag, mask) tuples for processing by zero padding.
    This function pads the bags to the maximum length in the batch and creates masks
    to indicate the real instances and padding.

    Args:
        batch: A list of tuples, where each tuple contains a case bag, control bag, and mask.

    Returns:
        A tuple containing padded case bags, padded control bags, case masks, and control masks.
    """
    durations: List[torch.Tensor] = [item[0] for item in batch]  # Extract events from the batch
    events: List[torch.Tensor] = [item[1] for item in batch]  # Extract durations from the batch
    traces_bags: List[torch.Tensor] = [item[2] for item in batch]  # Extract trace bags from the batch

    # Pad bags along the H dimension
    traces_padded_bags: torch.Tensor = pad_sequence(traces_bags, batch_first=True, padding_value=0.0)  # Shape [B, max_H, W]

    # Add the singleton channel dim back -> [B, 1, max_H, W]
    traces_padded_bags = traces_padded_bags.unsqueeze(1).to(torch.float32)

    # Create mask: 1 for real instances, 0 for padding
    traces_masks: torch.Tensor = torch.tensor(
        [[1] * bag.shape[0] + [0] * (traces_padded_bags.size(2) - bag.shape[0]) for bag in traces_bags]
    )

    return durations, events, traces_padded_bags, traces_masks

