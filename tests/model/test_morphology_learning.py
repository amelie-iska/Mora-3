"""Independent spatial fixture: a disconnected pixel path must fail this check.

Synthetic binary rectangles test learning only, never real-data coverage.
"""
import pytest
import numpy as np
import torch
from mora3.data.livecell import LiveCellExample
from mora3.data.livecell_graph import LiveCellGraphTask
from mora3.model.morphology import MorphologyForeground, foreground_objective


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shared_morphology_path_learns_pixel_aligned_signal(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
            torch.manual_seed(30)
            image = np.zeros((17, 19), dtype=np.uint8)
            image[3:14, 5:15] = 255
            example = LiveCellExample.from_coco(image=image,
                image_info=dict(id=1, file_name='A172_Phase_C7_1_00d00h00m_1.tif',
                                height=17, width=19),
                annotations=[], split='train', source_sha256='a'*64)
            task = LiveCellGraphTask(example, 'image_to_mask')
            # Hand-defined label, independent of polygon rasterizer/model.
            target = torch.zeros((1,17,19), dtype=torch.bool)
            target[0,3:14,5:15] = True
            target = target.to(device)
            model = MorphologyForeground(width=8).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            initial = foreground_objective(model(task), target).item()
            for _ in range(150):
                optimizer.zero_grad()
                loss = foreground_objective(model(task, for_training=True), target)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                logits = model(task)
                predicted = logits > 0
                truth = target[0]
                iou = (predicted & truth).sum() / (predicted | truth).sum()
                assert iou > .98
                assert foreground_objective(logits,target).item() < initial * .1
                # Input permutation breaks alignment while labels remain fixed.
                scrambled = np.flip(image, axis=0).copy()
                scrambled = np.roll(scrambled, 7, axis=1)
                from dataclasses import replace
                changed = LiveCellGraphTask(replace(example, pixels_hex=scrambled.tobytes().hex()),
                                            'image_to_mask')
                assert foreground_objective(model(changed), target).item() > .5
    finally:
        torch.set_num_threads(old_threads)
