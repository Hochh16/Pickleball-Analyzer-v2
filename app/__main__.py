"""Launch the setup wizard: `python -m app`.

Starts uvicorn and opens the default browser at the app URL. Host/port are
overridable via env (PB_APP_HOST / PB_APP_PORT) or --host/--port.
"""
from __future__ import annotations

import argparse
import os
import threading
import webbrowser

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Pickleball Analyzer v2 setup wizard")
    parser.add_argument("--host", default=os.environ.get("PB_APP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PB_APP_PORT", "8000")))
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not auto-open the browser.")
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload.")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser and not args.reload:
        # Open the browser shortly after the server starts accepting connections.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"\n  Pickleball Analyzer — setup wizard")
    print(f"  Open:  {url}\n")

    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
