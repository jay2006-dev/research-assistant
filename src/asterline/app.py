"""Streamlit UI entry point — stub for Milestone 1."""

import streamlit as st

from asterline.config import Settings


def main():
    settings = Settings()
    st.set_page_config(page_title=settings.app_name, layout="wide")
    st.title(settings.app_name)
    st.success("Application is running.")


if __name__ == "__main__":
    main()
