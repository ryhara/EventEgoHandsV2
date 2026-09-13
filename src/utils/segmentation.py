import cv2 as cv
import numpy as np
from .config import OUTPUT_HEIGHT, OUTPUT_WIDTH
import torch


def dilation_masks(masks, batch_size, kernel_size=5, itterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    masks = masks.detach().cpu().numpy().squeeze(1).astype(np.uint8)
    dilated_masks = [cv.dilate(mask, kernel, iterations=itterations) for mask in masks]
    dilated_masks = np.array(dilated_masks)
    return dilated_masks


def pc_normalize_batch(pc):
    pc[:, :, 0] /= OUTPUT_WIDTH
    pc[:, :, 1] /= OUTPUT_HEIGHT
    pc[:, :, :2] = 2 * pc[:, :, :2] - 1

    ts = pc[:, :, 2:]

    t_max = ts.max(1).values
    t_min = ts.min(1).values
    diff = t_max - t_min
    if torch.any(diff == 0):
        diff = t_max + 1e-6

    ts = 2 * ((ts - t_min.unsqueeze(1)) / (diff.unsqueeze(1))) - 1

    pc[:, :, 2:] = ts

    return pc


def mask_event_cloud_batch(masks_batch, events_batch, events_number_threshold=2048):
    filtered_events_list = [
        mask_event_cloud_one(mask, events, events_number_threshold)
        for mask, events in zip(masks_batch, events_batch)
    ]

    filtered_events_batch = torch.stack(filtered_events_list)

    return filtered_events_batch


def mask_event_cloud_one(mask, events, event_number_threshold=2048):
    # events: [event_count, 5], mask: [1, H, W]
    if events.shape[1] == 5:
        events_array = events.detach().cpu().numpy()
    else:
        events_array = events.detach().cpu().numpy().transpose(1, 0)

    mask_array = mask[0].cpu().numpy()

    x_coords = events_array[:, 0].astype(int)
    y_coords = events_array[:, 1].astype(int)
    valid_indices = (
        (y_coords >= 0) &
        (x_coords >= 0) &
        (mask_array[y_coords, x_coords] > 0)
    )
    filtered_events = events_array[valid_indices]

    if len(filtered_events) == 0:
        random_indices = np.random.choice(len(events_array), event_number_threshold)
        filtered_events = events_array[random_indices]
    elif len(filtered_events) < event_number_threshold:
        random_indices = np.random.choice(
            len(filtered_events), event_number_threshold - len(filtered_events)
        )
        filtered_events = np.concatenate((filtered_events, filtered_events[random_indices]))
    elif len(filtered_events) > event_number_threshold:
        random_indices = np.random.choice(len(filtered_events), event_number_threshold)
        filtered_events = filtered_events[random_indices]

    filtered_events_tensor = torch.tensor(filtered_events)
    return filtered_events_tensor


def bbox_event_one(bbox, events, event_number_threshold=2048):
    left_bbox = bbox["left"]
    right_bbox = bbox["right"]
    left_left, left_top, left_right, left_bottom = (
        left_bbox["left"],
        left_bbox["top"],
        left_bbox["right"],
        left_bbox["bottom"],
    )
    right_left, right_top, right_right, right_bottom = (
        right_bbox["left"],
        right_bbox["top"],
        right_bbox["right"],
        right_bbox["bottom"],
    )

    # events: [event_count, 5]
    events_array = events.detach().cpu().numpy()
    filtered_events = [
        evt
        for evt in events_array
        if (left_left <= evt[0] <= left_right and left_top <= evt[1] <= left_bottom)
        or (right_left <= evt[0] <= right_right and right_top <= evt[1] <= right_bottom)
        and (evt[3] > 0 or evt[4] > 0)
    ]

    if len(filtered_events) == 0:
        random_indices = np.random.choice(len(events_array), event_number_threshold)
        filtered_events = events_array[random_indices]
    elif len(filtered_events) < event_number_threshold:
        random_indices = np.random.choice(
            len(filtered_events), event_number_threshold - len(filtered_events)
        )
        random_events = [filtered_events[i] for i in random_indices]
        filtered_events.extend(random_events)
    elif len(filtered_events) > event_number_threshold:
        random_indices = np.random.choice(
            len(filtered_events), event_number_threshold, replace=False
        )
        filtered_events = [filtered_events[i] for i in random_indices]

    filtered_events_array = np.array(filtered_events)
    filtered_events_tensor = torch.tensor(filtered_events_array)

    return filtered_events_tensor
