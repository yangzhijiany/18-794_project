import numpy as np


# total_frame: target video frame
# ratio: our observe ratio, like 0.2 or 0.4
# num_frames: number of frames we want, default 16
# output: output frame length and numpy array of frames
def sample_frame(total_frames, ratio, num_frames=16):
    if not isinstance(total_frames, (int, np.integer)) or total_frames <= 0:
        raise ValueError("total_frames must be a positive integer")

    if not isinstance(num_frames, (int, np.integer)) or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")

    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")

    prefix_length = max(1, int(total_frames * ratio))

    # here, if prefix_length is lower than num_frames, duplication is possible
    frame_indices = np.linspace(
        0,
        prefix_length - 1,
        num_frames,
    ).astype(np.int64)

    return prefix_length, frame_indices