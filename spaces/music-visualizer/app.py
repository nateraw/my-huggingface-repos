from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import requests
import torch
from PIL import Image
from torchvision.io import write_video
from torchvision.transforms.functional import pil_to_tensor


def get_rgb_image(r=255, g=255, b=255, size=(1400, 900), overlay_im=None, return_pil=False):
    image = Image.new("RGBA", size, (r, g, b, 255))

    if overlay_im:
        img_w, img_h = overlay_im.size
        bg_w, bg_h = image.size
        offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
        image.alpha_composite(overlay_im, offset)
    image = image.convert("RGB")
    return image if return_pil else np.array(image)


def write_frames_between(image_a, image_b, out_dir="./images", n=500, skip_existing=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for i, t in enumerate(np.linspace(0.0, 1.0, n)):
        out_file = out_dir / f"image{i:06d}.jpg"
        if out_file.exists() and skip_existing:
            continue
        im_arr = torch.lerp(torch.tensor(image_a).float(), torch.tensor(image_b).float(), float(t))
        im = Image.fromarray(np.around(im_arr.numpy()).astype(np.uint8))
        im.save(out_file)


def get_timesteps_arr(audio_filepath, offset, duration, fps=30, margin=1.0, smooth=0.0):
    y, sr = librosa.load(audio_filepath, offset=offset, duration=duration)

    # librosa.stft hardcoded defaults...
    # n_fft defaults to 2048
    # hop length is win_length // 4
    # win_length defaults to n_fft
    D = librosa.stft(y, n_fft=2048, hop_length=2048 // 4, win_length=2048)

    # Extract percussive elements
    D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=margin)
    y_percussive = librosa.istft(D_percussive, length=len(y))

    # Get normalized melspectrogram
    spec_raw = librosa.feature.melspectrogram(y=y_percussive, sr=sr)
    spec_max = np.amax(spec_raw, axis=0)
    spec_norm = (spec_max - np.min(spec_max)) / np.ptp(spec_max)

    # Resize cumsum of spec norm to our desired number of interpolation frames
    x_norm = np.linspace(0, spec_norm.shape[-1], spec_norm.shape[-1])
    y_norm = np.cumsum(spec_norm)
    y_norm /= y_norm[-1]
    x_resize = np.linspace(0, y_norm.shape[-1], int(duration * fps))

    T = np.interp(x_resize, x_norm, y_norm)

    # Apply smoothing
    return T * (1 - smooth) + np.linspace(0.0, 1.0, T.shape[0]) * smooth


def make_fast_frame_video(
    frames_or_frame_dir="images",
    audio_filepath="music/thoughts.mp3",
    output_filepath="output.mp4",
    sr=44100,
    offset=7,
    duration=5,
    fps=30,
    margin=1.0,
    smooth=0.1,
    frame_filename_ext=".jpg",
):

    if isinstance(frames_or_frame_dir, list):
        frame_filepaths = frames_or_frame_dir
    else:
        frame_filepaths = sorted(Path(frames_or_frame_dir).glob(f"**/*{frame_filename_ext}"))

    num_frames = len(frame_filepaths)
    T = get_timesteps_arr(audio_filepath, offset, duration, fps=fps, margin=margin, smooth=smooth)
    yp = np.arange(num_frames)
    xp = np.linspace(0.0, 1.0, num_frames)

    frame_idxs = np.around(np.interp(T, xp, yp)).astype(np.int32)

    frames = None
    for img_path in [frame_filepaths[x] for x in frame_idxs]:
        frame = pil_to_tensor(Image.open(img_path)).unsqueeze(0)
        frames = frame if frames is None else torch.cat([frames, frame])
    frames = frames.permute(0, 2, 3, 1)

    y, sr = librosa.load(audio_filepath, sr=sr, mono=True, offset=offset, duration=duration)
    audio_tensor = torch.tensor(y).unsqueeze(0)

    write_video(
        output_filepath,
        frames,
        fps=fps,
        audio_array=audio_tensor,
        audio_fps=sr,
        audio_codec="aac",
        options={"crf": "23", "pix_fmt": "yuv420p"},
    )

    return output_filepath


OUTPUT_DIR = "multicolor_images_sm"
N = 500
IMAGE_SIZE = (640, 360)
MAX_DURATION = 10

if not Path(OUTPUT_DIR).exists():
    overlay_image_url = "https://huggingface.co/datasets/nateraw/misc/resolve/main/Group%20122.png"
    overlay_image = Image.open(requests.get(overlay_image_url, stream=True).raw, "r")
    hex_codes = ["#5e6179", "#ffbb9f", "#dfeaf2", "#75e9e5", "#ff6b6b"]

    rgb_vals = [tuple(int(hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)) for hex in hex_codes]

    for i, (rgb_a, rgb_b) in enumerate(zip(rgb_vals, rgb_vals[1:])):
        out_dir_step = Path(OUTPUT_DIR) / f"{i:06d}"
        image_a = get_rgb_image(*rgb_a, size=IMAGE_SIZE, overlay_im=overlay_image)
        image_b = get_rgb_image(*rgb_b, size=IMAGE_SIZE, overlay_im=overlay_image)
        write_frames_between(image_a, image_b, out_dir=out_dir_step, n=N)


def fn(audio_filepath):
    return make_fast_frame_video(
        OUTPUT_DIR,
        audio_filepath,
        "out.mp4",
        sr=44100,
        offset=0,
        duration=min(MAX_DURATION, librosa.get_duration(filename=audio_filepath)),
        fps=18,
    )


interface = gr.Interface(
    fn=fn,
    inputs=gr.Audio(type="filepath"),
    outputs="video",
    title="Music Visualizer",
    description="Create a simple music visualizer video with a cute 🤗 logo on top",
    article="<p style='text-align: center'><a href='https://github.com/nateraw/my-huggingface-repos/tree/main/spaces/music-visualizer' target='_blank'>Github Repo</a></p>",
    examples=[["https://huggingface.co/datasets/nateraw/misc/resolve/main/quick_example_loop.wav"]],
)

if __name__ == "__main__":
    interface.launch(debug=True)
