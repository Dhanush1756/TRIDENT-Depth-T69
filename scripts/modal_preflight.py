"""Read-only verification of existing Modal resources before CI deployment."""
import hashlib
from scripts.modal_checkpoint import SHA256, REMOTE, VOLUME_NAME

APP_NAME = "trident-depth-t69"
PUBLIC_URL = "https://dhanush1756--trident-depth-t69-web.us-east.modal.run"


def verify_existing_resources():
    import modal
    # Never create a replacement app/Volume, upload files or write checkpoint bytes.
    existing = modal.Function.from_name(APP_NAME, "web")
    if existing.get_web_url() != PUBLIC_URL:
        raise ValueError("existing Modal application URL differs; stop before deployment")
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    digest = hashlib.sha256()
    count = 0
    for chunk in volume.read_file(REMOTE):
        digest.update(chunk)
        count += len(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError("existing private checkpoint SHA256 differs; stop before deployment")
    return dict(status="PASS", app=APP_NAME, url=PUBLIC_URL,
                checkpoint_sha256=SHA256, checkpoint_bytes=count,
                private_checkpoint_written_to_disk=False, resources_created=False)


if __name__ == "__main__":
    import json
    print(json.dumps(verify_existing_resources(), indent=2))
