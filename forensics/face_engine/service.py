"""Compatibility entrypoint for running the WAYCON Face Engine service."""

from forensics.face_engine.app import app, create_app, main

__all__ = ["app", "create_app", "main"]


if __name__ == "__main__":
    main()
