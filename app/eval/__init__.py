"""Compatibility shim for ragas 0.2.* with modern langchain-community."""

from __future__ import annotations

import sys
import types

import langchain_community.chat_models

# ragas 0.2.* imports langchain_community.chat_models.vertexai.ChatVertexAI at
# module load time, but langchain-community >= 0.4 no longer ships that module.
# We only need the name to exist for isinstance checks; our eval code supplies
# its own LLM/embeddings wrappers, so the class is never instantiated.
if not hasattr(langchain_community.chat_models, "vertexai"):
    _vertexai_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    setattr(_vertexai_mod, "ChatVertexAI", type("ChatVertexAI", (), {}))
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_mod
    setattr(langchain_community.chat_models, "vertexai", _vertexai_mod)
