import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
from refacer import Refacer
import argparse
import ngrok
import imageio
import numpy as np
from PIL import Image
import tempfile
import base64
import pyfiglet
import shutil
import time

print("\033[94m" + pyfiglet.Figlet(font='slant').renderText("NeoRefacer") + "\033[0m")

def cleanup_temp(folder_path):
    try:
        shutil.rmtree(folder_path)
        print("Gradio cache cleared successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Prepare temp folder
os.environ["GRADIO_TEMP_DIR"] = "./tmp"
if os.path.exists("./tmp"):
    cleanup_temp(os.environ['GRADIO_TEMP_DIR'])
if not os.path.exists("./tmp"):
    os.makedirs("./tmp")

# Parse arguments
parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--max_num_faces", type=int, default=8)
parser.add_argument("--force_cpu", default=False, action="store_true")
parser.add_argument("--share_gradio", default=False, action="store_true")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=7860)
parser.add_argument("--colab_performance", default=False, action="store_true")
parser.add_argument("--ngrok", type=str, default=None)
parser.add_argument("--ngrok_region", type=str, default="us")
args = parser.parse_args()

# Initialize
refacer = Refacer(force_cpu=args.force_cpu, colab_performance=args.colab_performance)
num_faces = args.max_num_faces

def create_dummy_image():
    dummy = Image.new('RGB', (1, 1), color=(255, 255, 255))
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir="./tmp", suffix=".png")
    dummy.save(temp_file.name)
    return temp_file.name

def run_image(*vars):
    image_path = vars[0]
    origins = vars[1:(num_faces+1)]
    destinations = vars[(num_faces+1):(num_faces*2)+1]
    thresholds = vars[(num_faces*2)+1:-1]
    face_mode = vars[-1]

    disable_similarity = (face_mode in ["Single Face", "Multiple Faces"])
    multiple_faces_mode = (face_mode == "Multiple Faces")

    faces = []
    for k in range(num_faces):
        if destinations[k] is not None:
            faces.append({
                'origin': origins[k] if not multiple_faces_mode else None,
                'destination': destinations[k],
                'threshold': thresholds[k] if not multiple_faces_mode else 0.0
            })

    return refacer.reface_image(image_path, faces, disable_similarity=disable_similarity, multiple_faces_mode=multiple_faces_mode)

def run(*vars):
    video_path = vars[0]
    origins = vars[1:(num_faces+1)]
    destinations = vars[(num_faces+1):(num_faces*2)+1]
    thresholds = vars[(num_faces*2)+1:-2]
    preview = vars[-2]
    face_mode = vars[-1]

    disable_similarity = (face_mode in ["Single Face", "Multiple Faces"])
    multiple_faces_mode = (face_mode == "Multiple Faces")

    faces = []
    for k in range(num_faces):
        if destinations[k] is not None:
            faces.append({
                'origin': origins[k] if not multiple_faces_mode else None,
                'destination': destinations[k],
                'threshold': thresholds[k] if not multiple_faces_mode else 0.0
            })

    mp4_path, gif_path = refacer.reface(video_path, faces, preview=preview, disable_similarity=disable_similarity, multiple_faces_mode=multiple_faces_mode)
    return mp4_path, gif_path if gif_path else None

def load_first_frame(filepath):
    if filepath is None:
        return None
    frames = imageio.get_reader(filepath)
    return frames.get_data(0)

def extract_faces_auto(filepath, refacer_instance, max_faces=5, isvideo=False):
    if filepath is None:
        return [None] * max_faces

    # Check if video
    if isvideo:
        if os.path.getsize(filepath) > 5 * 1024 * 1024:  # larger than 5MB
            print("Video too large for auto-extract, skipping face extraction.")
            return [None] * max_faces

    frame = load_first_frame(filepath)
    if frame is None:
        return [None] * max_faces

    # Create manual temp image inside ./tmp
    temp_image_path = os.path.join("./tmp", f"temp_face_extract_{int(time.time() * 1000)}.png")
    Image.fromarray(frame).save(temp_image_path)

    try:
        faces = refacer_instance.extract_faces_from_image(temp_image_path, max_faces=max_faces)
        output_faces = faces + [None] * (max_faces - len(faces))
        return output_faces
    finally:
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_image_path}: {e}")

def toggle_tabs_and_faces(mode, face_tabs, origin_faces):
    if mode == "Single Face":
        tab_updates = [gr.update(visible=(i == 0)) for i in range(len(face_tabs))]
        origin_updates = [gr.update(visible=False) for _ in range(len(origin_faces))]
    elif mode == "Multiple Faces":
        tab_updates = [gr.update(visible=True) for _ in range(len(face_tabs))]
        origin_updates = [gr.update(visible=False) for _ in range(len(origin_faces))]
    else:
        tab_updates = [gr.update(visible=True) for _ in range(len(face_tabs))]
        origin_updates = [gr.update(visible=True) for _ in range(len(origin_faces))]
    return tab_updates + origin_updates

# --- UI ---

theme = gr.themes.Base(primary_hue="blue", secondary_hue="cyan")

with gr.Blocks(theme=theme, title="NeoRefacer - AI Refacer") as demo:
    with open("icon.png", "rb") as f:
        icon_data = base64.b64encode(f.read()).decode()
    icon_html = f'<img src="data:image/png;base64,{icon_data}" style="width:40px;height:40px;margin-right:10px;">'

    with gr.Row():
        gr.Markdown(f"""
        <div style="display: flex; align-items: center;">
        {icon_html}
        <span style="font-size: 2em; font-weight: bold; color:#2563eb;">NeoRefacer</span>
        </div>
        """)

    # --- IMAGE MODE ---
    with gr.Tab("Image Mode"):
        with gr.Row():
            image_input = gr.Image(label="Original image", type="filepath")
            image_output = gr.Image(label="Refaced image", interactive=False, type="filepath")

        with gr.Row():
            face_mode_image = gr.Radio(
                choices=["Single Face", "Multiple Faces", "Faces By Match"],
                value="Single Face",
                label="Replacement Mode"
            )
            image_btn = gr.Button("Reface Image", variant="primary")

        origin_image, destination_image, thresholds_image, face_tabs_image = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_image.append(origin)
            destination_image.append(destination)
            thresholds_image.append(threshold)
            face_tabs_image.append(tab)

        face_mode_image.change(
            fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_image, origin_image),
            inputs=[face_mode_image],
            outputs=face_tabs_image + origin_image
        )

        demo.load(
            fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_image, origin_image),
            inputs=None,
            outputs=face_tabs_image + origin_image
        )

        image_btn.click(
            fn=run_image,
            inputs=[image_input] + origin_image + destination_image + thresholds_image + [face_mode_image],
            outputs=[image_output]
        )

        image_input.change(
            fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces),
            inputs=image_input,
            outputs=origin_image
        )

    # --- GIF MODE ---
    with gr.Tab("GIF Mode"):
        with gr.Row():
            gif_input = gr.File(label="Original GIF", file_types=[".gif"])
            gif_preview = gr.Video(label="GIF Preview", interactive=False)
            gif_output = gr.Video(label="Refaced GIF (MP4)", interactive=False, format="mp4")
            gif_file_output = gr.Image(label="Refaced GIF (GIF)", type="filepath")

        with gr.Row():
            face_mode_gif = gr.Radio(
                choices=["Single Face", "Multiple Faces", "Faces By Match"],
                value="Single Face",
                label="Replacement Mode"
            )
            gif_btn = gr.Button("Reface GIF", variant="primary")

        preview_checkbox_gif = gr.Checkbox(label="Preview Generation (skip 90% of frames)", value=False)

        origin_gif, destination_gif, thresholds_gif, face_tabs_gif = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_gif.append(origin)
            destination_gif.append(destination)
            thresholds_gif.append(threshold)
            face_tabs_gif.append(tab)

        face_mode_gif.change(
            fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_gif, origin_gif),
            inputs=[face_mode_gif],
            outputs=face_tabs_gif + origin_gif
        )

        demo.load(
            fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_gif, origin_gif),
            inputs=None,
            outputs=face_tabs_gif + origin_gif
        )

        gif_btn.click(
            fn=lambda *args: run(*args),
            inputs=[gif_input] + origin_gif + destination_gif + thresholds_gif + [preview_checkbox_gif, face_mode_gif],
            outputs=[gif_output, gif_file_output]
        )

        gif_input.change(
            fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces),
            inputs=gif_input,
            outputs=origin_gif
        )
        gif_input.change(
            fn=lambda file: file,
            inputs=gif_input,
            outputs=[gif_preview]
        )

    # --- VIDEO MODE ---
    with gr.Tab("Video Mode"):
        with gr.Row():
            video_input = gr.Video(label="Original video", format="mp4")
            video_output = gr.Video(label="Refaced Video", interactive=False, format="mp4")

        with gr.Row():
            face_mode_video = gr.Radio(
                choices=["Single Face", "Multiple Faces", "Faces By Match"],
                value="Single Face",
                label="Replacement Mode"
            )
            video_btn = gr.Button("Reface Video", variant="primary")

        preview_checkbox_video = gr.Checkbox(label="Preview Generation (skip 90% of frames)", value=False)

        origin_video, destination_video, thresholds_video, face_tabs_video = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_video.append(origin)
            destination_video.append(destination)
            thresholds_video.append(threshold)
            face_tabs_video.append(tab)

        face_mode_video.change(
            fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_video, origin_video),
            inputs=[face_mode_video],
            outputs=face_tabs_video + origin_video
        )

        demo.load(
            fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_video, origin_video),
            inputs=None,
            outputs=face_tabs_video + origin_video
        )
        
        video_input.change(
            fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces, isvideo=True),
            inputs=video_input,
            outputs=origin_video
        )

        video_btn.click(
            fn=lambda *args: run(*args),
            inputs=[video_input] + origin_video + destination_video + thresholds_video + [preview_checkbox_video, face_mode_video],
            outputs=[video_output, gr.File(visible=False)]
        )

# --- ngrok connect (optional) ---
if args.ngrok:
    def connect(token, port, options):
        try:
            public_url = ngrok.connect(f"127.0.0.1:{port}", **options).url()
            print(f'ngrok URL: {public_url}')
        except Exception as e:
            print(f'ngrok connection aborted: {e}')

    connect(args.ngrok, args.server_port, {'region': args.ngrok_region, 'authtoken_from_env': False})

# --- Launch app ---
demo.queue().launch(favicon_path="icon.png", show_error=True, share=args.share_gradio, server_name=args.server_name, server_port=args.server_port)
