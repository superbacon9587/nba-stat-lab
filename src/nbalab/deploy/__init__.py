"""Everything needed to run the app on free public hosting.

- :mod:`manifest`: which tables and columns ship with the deployed app.
- :mod:`slim`: ``data/processed`` -> ``data/deploy`` (column-trimmed, zstd).
- :mod:`publish`: upload ``data/deploy`` to a Hugging Face dataset repo.
- :mod:`fetch`: download that dataset on first app start.
- :mod:`secrets`: read the Anthropic key from ``st.secrets``, env, or ``.env``.
- :mod:`rate_limit`: cap LLM-parsed questions per session and per app.
"""
