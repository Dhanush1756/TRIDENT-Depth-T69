"""Single-process application startup for local or container hosting."""
import os
import uvicorn


def main():
    port=int(os.environ.get("PORT","8000"))
    if not 1<=port<=65535:
        raise ValueError("PORT must be between 1 and 65535")
    uvicorn.run("backend.main:app",host=os.environ.get("TRIDENT_HOST","0.0.0.0"),port=port,workers=1)


if __name__=="__main__":main()
