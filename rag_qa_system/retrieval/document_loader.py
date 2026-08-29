from __future__ import annotations
import os
from typing import List, Dict, Any


class DocumentLoader:
    """
    Universal document loader supporting TXT, Markdown, Python, JSON, and PDF files.
    """

    @staticmethod
    def load_file(file_path: str) -> Dict[str, Any]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        if ext in [".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml", ".html", ".js", ".ts"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return {"source": filename, "content": content, "path": file_path, "type": ext[1:]}

        if ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
                return {"source": filename, "content": text, "path": file_path, "type": "pdf"}
            except Exception:
                with open(file_path, "rb") as f:
                    raw = f.read()
                return {"source": filename, "content": f"[PDF: {filename} ({len(raw)} bytes)]", "path": file_path, "type": "pdf"}

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return {"source": filename, "content": content, "path": file_path, "type": "text"}

    @staticmethod
    def load_directory(dir_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        docs = []
        if not os.path.exists(dir_path):
            return docs

        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.startswith(".") or f.endswith((".pyc", ".png", ".jpg", ".svg", ".ico", ".exe")):
                    continue
                fp = os.path.join(root, f)
                try:
                    docs.append(DocumentLoader.load_file(fp))
                except Exception:
                    pass
            if not recursive:
                break
        return docs
