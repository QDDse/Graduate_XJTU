import sys

sys.path.append("../")

import numpy as np
import pytest

from unittest.mock import patch, MagicMock
from dataset import TrainKaist  # Replace with the actual import


# Mock for the read_kaist function
def mock_read_kaist(path):
    # Return fake paths for testing that simulate the expected directory structure
    ir_img_paths = [
        "fake/path/to/kaist/lwir/image1.jpg",
        "fake/path/to/kaist/lwir/image2.jpg",
    ]
    vis_img_paths = [
        "fake/path/to/kaist/visible/image1.jpg",
        "fake/path/to/kaist/visible/image2.jpg",
    ]
    return ir_img_paths, vis_img_paths


@pytest.fixture
def args():
    # Create a mock of the arguments
    args = MagicMock()
    args.kaist_path = "/root/autodl-tmp/dataset/KAIST/"
    args.patch = 192
    return args


@pytest.fixture
def dataset(args):
    return TrainKaist(args)  # Replace with the actual class name if different


def test_len(dataset):
    # Test the __len__ method
    assert len(dataset) == 2  # Assuming the mock_read_kaist returns two pairs of images


def test_getitem(dataset):
    # Test the __getitem__ method
    # You can also mock cv2.imread and any other necessary functions here
    with patch("cv2.imread", return_value=np.zeros((256, 256, 3), dtype=np.uint8)):
        ir_img, vis_img = dataset[0]
        assert ir_img.shape == (
            1,
            dataset.patch,
            dataset.patch,
        )  # Check that the shape is correct after cropping and permutation
        assert vis_img.shape == (1, dataset.patch, dataset.patch)


def test_crops(dataset):
    # Test the crops method
    fake_ir_img = np.zeros((256, 256, 1), dtype=np.float32)
    fake_vis_img = np.zeros((256, 256, 1), dtype=np.float32)
    cropped_ir_img, cropped_vis_img = dataset.crops(fake_ir_img, fake_vis_img)
    assert cropped_ir_img.shape == (dataset.patch, dataset.patch, 1)
    assert cropped_vis_img.shape == (dataset.patch, dataset.patch, 1)


# Add more tests as necessary
