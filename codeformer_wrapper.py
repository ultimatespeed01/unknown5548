# codeformer_wrapper.py
# Copyright (c) 2022 Shangchen Zhou
# Modifications and additions copyright (c) 2025 Felipe Daragon

# License: CC BY-NC-SA 4.0 (https://github.com/felipedaragon/codeformer/blob/main/README.md)
# Same as the original code by Shangchen Zhou.

import os
import torch
import cv2
from pathlib import Path
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY

# Prepare device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CodeFormer model once
pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                       connect_list=['32', '64', '128', '256']).to(device)

ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                                model_dir='weights/CodeFormer', progress=True, file_name=None)
checkpoint = torch.load(ckpt_path)['params_ema']
net.load_state_dict(checkpoint)
net.eval()

# Load helper
face_helper = FaceRestoreHelper(
    upscale_factor=1,  # No background upscaling
    face_size=512,
    crop_ratio=(1, 1),
    det_model='retinaface_resnet50',
    save_ext='jpg',
    use_parse=True,
    device=device
)

def enhance_image(input_image_path: str, w: float = 0.5) -> str:
    """
    Enhances an input image using CodeFormer and saves it with a '.enhanced.jpg' suffix.

    Args:
        input_image_path (str): Path to the input image (JPG or PNG).
        w (float): Balance quality and fidelity (default=0.5).

    Returns:
        str: Path to the enhanced image.
    """
    input_path = Path(input_image_path)
    output_path = input_path.with_name(f"{input_path.stem}.enhanced.jpg")

    # Clean previous state
    face_helper.clean_all()

    # Load image
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {input_image_path}")

    face_helper.read_image(img)
    num_faces = face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    if num_faces == 0:
        raise ValueError(f"No faces detected in: {input_image_path}")

    face_helper.align_warp_face()

    # Enhance each face
    for cropped_face in face_helper.cropped_faces:
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        with torch.no_grad():
            output = net(cropped_face_t, w=w, adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face)

    # Paste faces back
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()

    # Save output
    os.makedirs(output_path.parent, exist_ok=True)
    cv2.imwrite(str(output_path), restored_img)

    print(f"Enhanced image saved to: {output_path}")
    return str(output_path)
