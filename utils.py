"""FasterGSBasis/utils.py"""

import io
import warnings
import contextlib

import torch


def enable_expandable_segments() -> bool:
    """Return True if 'expandable_segments' allocator feature is available on this device."""
    torch.cuda.memory._set_allocator_settings('expandable_segments:True')
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stderr(stderr_buffer), warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always')
        torch.empty(1, device='cuda')  # allocate gpu memory to trigger potential warning
    stderr_output = stderr_buffer.getvalue()
    for warning in caught_warnings:
        if 'expandable_segments' in str(warning.message):
            return False
    if 'expandable_segments' in stderr_output:
        return False
    return True
