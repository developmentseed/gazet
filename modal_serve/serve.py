"""Modal deployment for Gazet.

Three independently scaled containers:
- LlamaServer: GPU T4 running the official llama.cpp server image
- Api: CPU container serving gazet.api as a native ASGI app
- Demo: CPU container running the Streamlit UI

Deploy:
    modal deploy modal_serve/serve.py
"""

from __future__ import annotations

import modal

APP_NAME = "gazet"

MODEL_PATH = "/models/checkpoints/qwen35-fientune-v3/ckpt-q4_k_m.gguf"
DATA_PATH = "/data"

app = modal.App(APP_NAME)

gazet_vol = modal.Volume.from_name("gazet")
data_vol = modal.Volume.from_name("gazet-data")


# Llama Server
# The base image's ENTRYPOINT is ["/app/llama-server"], which is incompatible
# with Modal's runtime: per Modal docs, an image's ENTRYPOINT must exec args
# passed to it so the Python harness can launch. We override it to a no-op
# passthrough, relocate libs to a canonical path, and ldconfig them.
llama_image = (
    modal.Image.from_registry(
        "ghcr.io/ggml-org/llama.cpp:server-cuda",
        add_python="3.11",
    )
    .entrypoint([])
    .run_commands(
        "cp -r /app /usr/local/lib/llama",
        "echo /usr/local/lib/llama > /etc/ld.so.conf.d/llama.conf",
        "ldconfig",
    )
    .env({"LD_LIBRARY_PATH": "/usr/local/lib/llama"})
)


@app.cls(
    image=llama_image,
    gpu="T4",
    volumes={"/models": gazet_vol},
    scaledown_window=300,
    min_containers=1,
    max_containers=2,
    timeout=600,
)
@modal.concurrent(max_inputs=4)
class LlamaServer:
    @modal.web_server(port=9000, startup_timeout=120)
    def serve(self):
        import subprocess

        subprocess.Popen(
            [
                "/usr/local/lib/llama/llama-server",
                "-m", MODEL_PATH,
                "-ngl", "99",
                "--host", "0.0.0.0",
                "--port", "9000",
                "--ctx-size", "2048",
            ]
        )


# FastAPI app
api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .add_local_python_source("gazet")
)


@app.cls(
    image=api_image,
    volumes={"/data": data_vol},
    scaledown_window=300,
    min_containers=1,
    max_containers=3,
    timeout=300,
)
@modal.concurrent(max_inputs=10)
class Api:
    @modal.enter()
    def setup(self):
        import os

        os.environ.setdefault("GAZET_DATA_DIR", DATA_PATH)
        os.environ.setdefault("GAZET_USE_NORMALIZED_DATA", "1")
        os.environ["LLAMA_SERVER_URL"] = LlamaServer().serve.get_web_url()

    @modal.asgi_app()
    def fastapi_app(self):
        from gazet.api import app as fastapi_app

        return fastapi_app


# Streamlit Demo
demo_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["demo"])
    .add_local_python_source("gazet")
    .add_local_file("gazet_demo.py", "/root/gazet_demo.py")
    .add_local_file("assets/gazet-logo.svg", "/root/assets/gazet-logo.svg")
    .add_local_file("assets/ds-logo-pos.svg", "/root/assets/ds-logo-pos.svg")
)


@app.cls(
    image=demo_image,
    scaledown_window=600,
    min_containers=1,
    max_containers=3,
    timeout=600,
)
@modal.concurrent(max_inputs=10)
class Demo:
    @modal.web_server(port=7860, startup_timeout=60)
    def serve(self):
        import os
        import subprocess

        env = os.environ.copy()
        env["GAZET_API_URL"] = Api().fastapi_app.get_web_url()

        subprocess.Popen(
            [
                "streamlit", "run", "/root/gazet_demo.py",
                "--server.port", "7860",
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false",
            ],
            env=env,
        )
