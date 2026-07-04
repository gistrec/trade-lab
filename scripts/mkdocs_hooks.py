"""MkDocs build hook: inject repo-root research markdown into the site.

`findings/*.md` and `RESULTS.md` live at the repo ROOT — `findings/` is
immutable history referenced by RESULTS.md, `tests/`, `src/`, and CLAUDE.md,
so it must NOT be moved into `docs/`. MkDocs only reads its `docs_dir`, so this
``on_files`` hook appends ``File`` objects that read those root files in place
and write them under ``site/``. Output goes only to ``site/`` (gitignored), so
the working tree stays clean.

Injected pages are addressable in the manual ``nav`` by their src path
(``findings/<name>.md``, ``RESULTS.md``) — the same pattern mkdocs-gen-files
and mkdocstrings rely on.
"""
from __future__ import annotations

import glob
import os

from mkdocs.structure.files import File


def _repo_root(config) -> str:
    # docs_dir is <repo>/docs; its parent is the repo root.
    return os.path.dirname(os.path.abspath(config["docs_dir"]))


def on_files(files, config):
    root = _repo_root(config)
    use_dir_urls = config["use_directory_urls"]
    site_dir = config["site_dir"]

    def inject(rel_path: str) -> None:
        files.append(File(
            path=rel_path,
            src_dir=root,
            dest_dir=site_dir,
            use_directory_urls=use_dir_urls,
        ))

    # Master index at the repo root.
    if os.path.exists(os.path.join(root, "RESULTS.md")):
        inject("RESULTS.md")

    # Every research writeup under findings/ (immutable, read in place).
    for abs_path in sorted(glob.glob(os.path.join(root, "findings", "*.md"))):
        inject(os.path.relpath(abs_path, root))

    return files
