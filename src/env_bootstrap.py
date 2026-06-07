"""
env_bootstrap.py — Mirror Streamlit Cloud secrets into os.environ.

Streamlit exposes secrets via st.secrets; config.py reads os.getenv at import
time. This helper runs before src imports in streamlit_app.py so HF and DB
credentials are visible to the rest of the stack.
"""

from __future__ import annotations

import os


def apply_streamlit_secrets() -> None:
    """Copy top-level st.secrets string entries into os.environ."""
    try:
        import streamlit as st  # noqa: PLC0415
    except ImportError:
        return

    try:
        for key, value in st.secrets.items():
            if isinstance(value, str) and value.strip():
                os.environ[key] = value.strip()
    except Exception:
        return
