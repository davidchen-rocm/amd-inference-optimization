"""Run the private service with exactly one supervisor and a loopback listener."""

import uvicorn

from .api import create_app


def main():
    uvicorn.run(
        create_app(),
        host="127.0.0.1",
        port=8791,
        workers=1,
        access_log=False,
        proxy_headers=False,
        limit_concurrency=32,
        timeout_keep_alive=5,
        server_header=False,
    )


if __name__ == "__main__":
    main()
