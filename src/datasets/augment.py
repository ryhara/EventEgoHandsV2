import numpy as np


def augment_shuffle_polarity(
    events: np.ndarray, probability: float = 0.5
) -> np.ndarray:
    # x, y, t, p
    if events.size == 0 or events.ndim == 1 or events.shape[1] < 4:
        return events
    augment = np.random.rand(events.shape[0]) < probability
    # p = 0 or 1
    events[augment, -1] = np.random.randint(0, 2, np.sum(augment))
    return events


def augment_events_flip(events: np.ndarray) -> np.ndarray:
    # x, y, t, p
    if events.size == 0 or events.ndim == 1 or events.shape[1] < 4:
        return events
    events[:, -1] = np.abs(1 - events[:, -1])
    return events


def augment_events_time_shuffle(
    events: np.ndarray, probability: float = 0.5
) -> np.ndarray:
    # x, y, t, p
    if events.size == 0 or events.ndim == 1 or events.shape[1] < 4:
        return events
    augment = np.random.rand(events.shape[0]) < probability
    shuffled_indices = np.random.permutation(np.sum(augment))
    events[augment, 2] = events[augment, 2][shuffled_indices]
    return events
