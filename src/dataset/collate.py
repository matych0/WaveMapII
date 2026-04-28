from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


""" def collate_padding(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return traces_padded_bags, traces_mask, durations, events """


def collate_padding(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    
    traces_bags = [item["traces"] for item in batch]
    durations = torch.stack([item["duration"] for item in batch])
    events = torch.stack([item["event"] for item in batch])

    study_ids = [item["study_id"] for item in batch]
    center_ids = [item["center_id"] for item in batch]

    # Create y = [event, duration]
    y = torch.stack([events, durations], dim=1)  # Shape [B, 2]

    # Pad bags
    traces_padded_bags = pad_sequence(
        traces_bags,
        batch_first=True,
        padding_value=0.0
    )  # [B, max_H, W]

    traces_padded_bags = traces_padded_bags.unsqueeze(1).to(torch.float32)

    # Mask
    max_len = traces_padded_bags.size(2)
    traces_mask = torch.zeros((len(traces_bags), max_len), dtype=torch.float32)

    for i, bag in enumerate(traces_bags):
        traces_mask[i, :bag.shape[0]] = 1.0

    return {
        "traces": traces_padded_bags,
        "mask": traces_mask,
        "y": y,
        "study_id": study_ids,
        "center_id": center_ids
    }


def collate_padding_merged(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Collates a Single_control sampled batch of (case_bag, control_bag) pairs for processing by zero padding.
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
    """ Collates a batch of (case_bag, control_bag) tuples separately for processing by zero padding. """
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
    """ Collates a batch for inference by zero padding."""
    durations: List[torch.Tensor] = [item[0] for item in batch]
    events: List[torch.Tensor] = [item[1] for item in batch]
    traces_bags: List[torch.Tensor] = [item[2] for item in batch]
    traces_orig: List[torch.Tensor] = [item[3] for item in batch]
    filepaths: List[str] = [item[4] for item in batch]
    trace_indices: List[int] = [item[5] for item in batch]
    ptp: List[torch.Tensor] = [item[6] for item in batch]

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
    """ Collates a Single-control sampled batch of (duration, event, trace_bag) tuples for validation by zero padding."""
    durations: List[torch.Tensor] = [item[0] for item in batch]
    events: List[torch.Tensor] = [item[1] for item in batch]
    traces_bags: List[torch.Tensor] = [item[2] for item in batch]

    # Pad bags along the H dimension
    traces_padded_bags: torch.Tensor = pad_sequence(traces_bags, batch_first=True, padding_value=0.0)  # Shape [B, max_H, W]

    # Add the singleton channel dim back -> [B, 1, max_H, W]
    traces_padded_bags = traces_padded_bags.unsqueeze(1).to(torch.float32)

    # Create mask: 1 for real instances, 0 for padding
    traces_masks: torch.Tensor = torch.tensor(
        [[1] * bag.shape[0] + [0] * (traces_padded_bags.size(2) - bag.shape[0]) for bag in traces_bags]
    )

    return durations, events, traces_padded_bags, traces_masks


""" def collate_patches(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(batch)
    CH = batch[0][0].shape[1]
    L = batch[0][0].shape[2]

    # number of patches per individual
    P_list = [item[0].shape[0] for item in batch]
    P_max = max(P_list)

    # allocate tensors
    traces_padded = torch.zeros(B, P_max, CH, L)
    masks = torch.zeros(B, P_max, dtype=torch.bool)

    durations = torch.zeros(B)
    events = torch.zeros(B)

    # fill tensors
    for i, (traces, duration, event) in enumerate(batch):
        Pi = traces.shape[0]

        traces_padded[i, :Pi] = traces
        masks[i, :Pi] = 1

        durations[i] = duration
        events[i] = event

    # reshape for CNN
    traces_flat = traces_padded.view(B * P_max, CH, L)

    events = events.to(torch.bool)

    return traces_flat, masks, durations, events """


def collate_patches(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collates a batch of dict samples with patch structure and zero padding."""

    B = len(batch)
    CH = batch[0]["traces"].shape[1]
    L = batch[0]["traces"].shape[2]

    study_ids = [item["study_id"] for item in batch]
    center_ids = [item["center_id"] for item in batch]

    # number of patches per individual
    P_list = [item["traces"].shape[0] for item in batch]
    P_max = max(P_list)

    # allocate tensors
    traces_padded = torch.zeros(B, P_max, CH, L)
    masks = torch.zeros(B, P_max, dtype=torch.bool)

    durations = torch.zeros(B)
    events = torch.zeros(B)

    # fill tensors
    for i, item in enumerate(batch):
        traces = item["traces"]
        duration = item["duration"]
        event = item["event"]

        Pi = traces.shape[0]

        traces_padded[i, :Pi] = traces
        masks[i, :Pi] = 1

        durations[i] = duration
        events[i] = event

    # reshape for CNN
    traces_flat = traces_padded.view(B * P_max, CH, L)

    # 👇 IMPORTANT: unify into y
    events = events.float()  # must match dtype with durations
    y = torch.stack([events, durations], dim=1)  # [B, 2]

    return {
        "traces": traces_flat,   # [B * P_max, CH, L]
        "mask": masks,           # [B, P_max]
        "y": y,                   # [B, 2]
        "study_id": study_ids,
        "center_id": center_ids
    }


def collate_amplitudes(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Collates a batch of (voltages, duration, event) tuples for processing by zero padding. """
    voltages: List[torch.Tensor] = [item[0] for item in batch]
    durations: torch.Tensor = torch.stack([item[1] for item in batch])
    events: torch.Tensor = torch.stack([item[2] for item in batch])

    # Pad voltages along the H dimension
    voltages_padded: torch.Tensor = pad_sequence(voltages, batch_first=True, padding_value=0.0)  # Shape [B, max_H]

    # Create mask: 1 for real instances, 0 for padding
    masks = torch.tensor(
        [[1] * voltage.shape[0] + [0] * (voltages_padded.shape[1] - voltage.shape[0]) for voltage in voltages]
    )

    return voltages_padded, masks, durations, events
    