"""Render a real animated ENDING shot from the film's final frame (no freeze).

Uploads output last frame -> ComfyUI /input, animates with WAN 2.2 i2v (gentle
sunset gaze) so the closing ~13s has genuine motion instead of a frozen hold.
"""
import io, json, sys, urllib.request
from comfymovies.film import (
    FilmSpec, ANIME_NEGATIVE, ANIME_MOTION_ANCHOR,
    UNET_HIGH, UNET_LOW, WAN_CLIP, WAN_VAE, WAN_SHIFT,
)
from comfymovies.wan import wan_frames
from comfymovies.comfy import ComfyClient

HOST, PORT = "192.168.1.90", 8188
FRAME = "/tmp/ff_lastframe.png"
NAME = "ff_ending.png"
W, H, FPS = 1280, 720, 16
SECONDS = 13.0
SEED = 77
STEPS, CFG = 20, 3.5

CHARACTER = ("A young girl about ten years old with a short black bob and a small "
             "red hair ribbon, big curious dark eyes, wearing a pale blue dress with "
             "white collar and red rain boots, beside a small round fluffy white forest "
             "spirit with two big black eyes, rosy cheeks and two tiny green leaf ears")
MOTION = ("the girl and the little white spirit sit close together on the hilltop at "
          "golden hour, gazing out at the glowing sunset valley, their hair and the "
          "grass swaying softly in a warm breeze, fireflies and dust motes drifting "
          "through the god-rays, the spirit's fur ruffling gently, clouds slowly "
          "moving, a calm tender resolving moment, very slow cinematic pull-back")


def upload(path, name):
    data = open(path, "rb").read()
    boundary = "----ffend"
    body = io.BytesIO()
    def w(s): body.write(s.encode() if isinstance(s, str) else s)
    w(f"--{boundary}\r\n")
    w(f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n')
    w("Content-Type: image/png\r\n\r\n"); w(data); w("\r\n")
    w(f"--{boundary}\r\n")
    w('Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n')
    w(f"--{boundary}--\r\n")
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/upload/image", data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    print("upload:", json.loads(urllib.request.urlopen(req).read()))


def graph():
    length = wan_frames(SECONDS, FPS)
    boundary = max(1, min(STEPS - 1, round(STEPS * 0.5)))
    g = {}
    def node(nid, ct, inp): g[nid] = {"class_type": ct, "inputs": inp}; return nid
    node("img", "LoadImage", {"image": NAME})
    node("scale", "ImageScale", {"image": ["img", 0], "width": W, "height": H,
                                 "upscale_method": "lanczos", "crop": "center"})
    node("wclip", "CLIPLoader", {"clip_name": WAN_CLIP, "type": "wan", "device": "default"})
    node("wvae", "VAELoader", {"vae_name": WAN_VAE})
    node("whi", "UNETLoader", {"unet_name": UNET_HIGH, "weight_dtype": "default"})
    node("wlo", "UNETLoader", {"unet_name": UNET_LOW, "weight_dtype": "default"})
    node("mshi", "ModelSamplingSD3", {"model": ["whi", 0], "shift": WAN_SHIFT})
    node("mslo", "ModelSamplingSD3", {"model": ["wlo", 0], "shift": WAN_SHIFT})
    node("wpos", "CLIPTextEncode",
         {"text": f"{CHARACTER}. {MOTION}. {ANIME_MOTION_ANCHOR}", "clip": ["wclip", 0]})
    node("wneg", "CLIPTextEncode", {"text": ANIME_NEGATIVE, "clip": ["wclip", 0]})
    node("i2v", "WanImageToVideo", {
        "positive": ["wpos", 0], "negative": ["wneg", 0], "vae": ["wvae", 0],
        "width": W, "height": H, "length": length, "batch_size": 1,
        "start_image": ["scale", 0]})
    node("khi", "KSamplerAdvanced", {
        "model": ["mshi", 0], "add_noise": "enable", "noise_seed": SEED,
        "steps": STEPS, "cfg": CFG, "sampler_name": "euler", "scheduler": "simple",
        "positive": ["i2v", 0], "negative": ["i2v", 1], "latent_image": ["i2v", 2],
        "start_at_step": 0, "end_at_step": boundary, "return_with_leftover_noise": "enable"})
    node("klo", "KSamplerAdvanced", {
        "model": ["mslo", 0], "add_noise": "disable", "noise_seed": SEED,
        "steps": STEPS, "cfg": CFG, "sampler_name": "euler", "scheduler": "simple",
        "positive": ["i2v", 0], "negative": ["i2v", 1], "latent_image": ["khi", 0],
        "start_at_step": boundary, "end_at_step": STEPS, "return_with_leftover_noise": "disable"})
    node("dec", "VAEDecode", {"samples": ["klo", 0], "vae": ["wvae", 0]})
    node("vid", "CreateVideo", {"images": ["dec", 0], "fps": float(FPS)})
    node("save", "SaveVideo", {"video": ["vid", 0],
                               "filename_prefix": "ComfyUIMovies/ffend",
                               "format": "auto", "codec": "auto"})
    return g


if __name__ == "__main__":
    upload(FRAME, NAME)
    c = ComfyClient(host=HOST, port=PORT)
    pid = c.submit(graph())
    print("queued ending ->", pid)
    json.dump({"end": pid}, open("/tmp/ffend_pid.json", "w"))
