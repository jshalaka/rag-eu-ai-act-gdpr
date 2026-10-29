"""Streamlit Community Cloud entrypoint.

Set the app's main file to `cloud/streamlit_app.py`. There is no build step on that platform, so
the first start builds the search index from the PDFs in `data/raw` (about a minute) and keeps it in
memory and in a temporary folder. It is rebuilt whenever the app wakes up from sleep.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # the repo uses a src layout and is not pip-installed

from rag_eu_law.app.bootstrap import ensure_index  # noqa: E402
from rag_eu_law.app.cloud_env import apply_cloud_environment  # noqa: E402
from rag_eu_law.app.config import Settings  # noqa: E402
from rag_eu_law.app.container import Container  # noqa: E402
from rag_eu_law.app.ui import configure_page, main  # noqa: E402
from rag_eu_law.domain.exceptions import ConfigurationError  # noqa: E402

WORK_DIR = Path(tempfile.gettempdir()) / "rag-eu-law"


def secret_names() -> list[str]:
    """Names (never values) of the secrets the platform gave this app."""
    try:
        return sorted(str(key) for key in st.secrets)
    except FileNotFoundError:
        return []


def show_configuration_error(error: ConfigurationError) -> None:
    """Streamlit hides exception text on the page, so print the (value-free) message ourselves."""
    st.error(f"Configuration problem: {error}")
    names = secret_names()
    st.markdown(
        "Secrets must be **top-level** entries named exactly like the `.env` keys, for example "
        '`RAG_OPENAI_API_KEY = "sk-..."`, outside any `[section]`. '
        + (
            "Secret names the app can see: " + ", ".join(f"`{name}`" for name in names)
            if names
            else "The app can see **no secrets at all**: open the app's **Settings → Secrets**."
        )
    )
    st.stop()


def apply_environment() -> None:
    """Copy platform secrets into the environment and lock the cloud-specific settings."""
    try:
        secrets = dict(st.secrets.items())
    except FileNotFoundError:  # no secrets configured (local run): rely on the environment
        secrets = {}
    apply_cloud_environment(secrets, os.environ, REPO_ROOT, WORK_DIR)


@st.cache_resource(
    show_spinner="Building the search index. The first start takes about a minute..."
)
def prepared_container() -> Container:
    apply_environment()
    container = Container(Settings(_env_file=None))
    ensure_index(container)
    return container


try:
    main(prepared_container)
except ConfigurationError as error:
    configure_page()
    show_configuration_error(error)
