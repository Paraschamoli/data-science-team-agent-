"""Data loading utilities for various file formats."""

import os
from pathlib import Path

import pandas as pd
from langchain.tools import tool  # type: ignore[import]

ALLOW_UNSAFE_PICKLE_ENV_VAR = "ALLOW_UNSAFE_PICKLE"
DEFAULT_MAX_MB = 20  # cap file size we attempt to load
DEFAULT_MAX_ROWS = 5000  # cap rows read per file to avoid OOM
DEFAULT_MAX_ENTRIES = 1000  # cap directory recursion output
DEFAULT_MAX_DEPTH = 5  # cap directory recursion depth


def _pickle_loading_allowed() -> bool:
    """Check if pickle loading is allowed.

    Pickle deserialization executes arbitrary code.
    This helper enforces an explicit opt-in via env var to avoid RCE on untrusted data.

    Returns:
        bool: True if pickle loading is allowed, False otherwise.

    """
    return os.getenv(ALLOW_UNSAFE_PICKLE_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


@tool(response_format="content_and_artifact")
def load_directory(
    directory_path: str = os.getcwd(),
    file_type: str | None = None,
    max_mb: int = DEFAULT_MAX_MB,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> tuple[str, dict]:
    """Load all recognized tabular files in a directory into memory.

    If you only need filenames (a directory listing), use
    `list_directory_contents` or `search_files_by_pattern` instead.
    If file_type is specified (e.g., 'csv'), only files
    with that extension are loaded.

    Args:
        directory_path: The path to the directory to load. Defaults to the current working directory.
        file_type: Optional file type filter (e.g., 'csv', 'xlsx').
        max_mb: Maximum file size in MB to attempt loading.
        max_rows: Maximum number of rows to read per file.

    Returns:
        Tuple containing status message and data dictionary.

    """
    print(f"    * Tool: load_directory | {directory_path}")

    import os

    if directory_path is None:
        return "No directory path provided.", {}

    try:
        base_path = Path(directory_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid directory path: {exc}", {}

    if not base_path.is_dir():
        return f"Directory not found: {base_path}", {}

    data_frames: dict[str, dict] = {}
    max_bytes = max_mb * 1024 * 1024 if max_mb else None

    for filename in sorted(os.listdir(base_path)):
        file_path = base_path / filename

        # Skip directories
        if file_path.is_dir():
            continue

        # If file_type is specified, only process files that match.
        if file_type and not filename.lower().endswith(f".{file_type.lower()}"):
            continue

        if max_bytes is not None and file_path.stat().st_size > max_bytes:
            data_frames[filename] = {
                "status": "skipped",
                "data": None,
                "error": f"Skipped: file larger than {max_mb}MB",
            }
            continue

        try:
            df, error = _load_file_safe(file_path, max_rows)
            if error:
                data_frames[filename] = {
                    "status": "error",
                    "data": None,
                    "error": error,
                }
            else:
                data_frames[filename] = {
                    "status": "loaded",
                    "data": df.to_dict() if df is not None else None,
                    "shape": df.shape if df is not None else None,
                }
        except Exception as exc:
            data_frames[filename] = {
                "status": "error",
                "data": None,
                "error": f"Error loading {filename}: {exc}",
            }

    message = f"Loaded {len([f for f in data_frames.values() if f['status'] == 'loaded'])} files from {directory_path}"
    return message, data_frames


@tool(response_format="content_and_artifact")
def load_file(
    file_path: str,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> tuple[str, dict]:
    """Load a single recognized tabular file into memory.

    Args:
        file_path: The path to the file to load.
        max_rows: Maximum number of rows to read. Defaults to 5000.

    Returns:
        A tuple containing a message and the loaded data.

    """
    print(f"    * Tool: load_file | {file_path}")

    try:
        path = Path(file_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid file path: {exc}", {}

    if not path.is_file():
        return f"File not found: {path}", {}

    try:
        df, error = _load_file_safe(path, max_rows)
        if error or df is None:
            return f"Error loading file: {error}", {}
        else:
            message = f"Successfully loaded {path.name} with shape {df.shape}"
            return message, {"data": df.to_dict(), "shape": df.shape}
    except Exception as exc:
        return f"Error loading {file_path}: {exc}", {}


@tool
def list_directory_contents(
    directory_path: str = os.getcwd(),
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> str:
    """List the contents of a directory without loading any files.

    Args:
        directory_path: The path to the directory to list. Defaults to the current working directory.
        max_entries: Maximum number of entries to return. Defaults to 1000.

    Returns:
        A formatted listing of directory contents.

    """
    print(f"    * Tool: list_directory_contents | {directory_path}")

    try:
        base_path = Path(directory_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid directory path: {exc}"

    if not base_path.is_dir():
        return f"Directory not found: {base_path}"

    try:
        entries = []
        for entry in sorted(os.listdir(base_path))[:max_entries]:
            entry_path = base_path / entry
            if entry_path.is_dir():
                entries.append(f"📁 {entry}/")
            else:
                size = entry_path.stat().st_size
                entries.append(f"📄 {entry} ({size:,} bytes)")

        return "\n".join(entries)
    except Exception as exc:
        return f"Error listing directory: {exc}"


@tool
def list_directory_recursive(
    directory_path: str = os.getcwd(),
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> str:
    """Recursively list directory contents up to a maximum depth.

    Args:
        directory_path: The path to the directory to list. Defaults to the current working directory.
        max_depth: Maximum recursion depth. Defaults to 5.
        max_entries: Maximum number of entries to return. Defaults to 1000.

    Returns:
        A formatted recursive listing of directory contents.

    """
    print(f"    * Tool: list_directory_recursive | {directory_path}")

    try:
        base_path = Path(directory_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid directory path: {exc}"

    if not base_path.is_dir():
        return f"Directory not found: {base_path}"

    def _list_recursive(path, depth, entries):
        if depth > max_depth or len(entries) >= max_entries:
            return

        try:
            for entry in sorted(os.listdir(path)):
                if len(entries) >= max_entries:
                    break

                entry_path = path / entry
                indent = "  " * depth

                if entry_path.is_dir():
                    entries.append(f"{indent}📁 {entry}/")
                    _list_recursive(entry_path, depth + 1, entries)
                else:
                    size = entry_path.stat().st_size
                    entries.append(f"{indent}📄 {entry} ({size:,} bytes)")
        except (OSError, PermissionError):
            pass

    entries = []
    _list_recursive(base_path, 0, entries)

    return "\n".join(entries)


@tool
def get_file_info(file_path: str) -> str:
    """Get detailed information about a specific file.

    Args:
        file_path: The path to the file to get information about.

    Returns:
        Detailed file information.

    """
    print(f"    * Tool: get_file_info | {file_path}")

    try:
        path = Path(file_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid file path: {exc}"

    if not path.is_file():
        return f"File not found: {path}"

    try:
        stat = path.stat()
        info = [
            f"File: {path.name}",
            f"Size: {stat.st_size:,} bytes",
            f"Modified: {pd.Timestamp.fromtimestamp(stat.st_mtime)}",
            f"Type: {path.suffix or 'no extension'}",
        ]

        # Try to detect if it's a tabular file
        tabular_extensions = {".csv", ".xlsx", ".xls", ".parquet", ".json", ".tsv"}
        if path.suffix.lower() in tabular_extensions:
            info.append("Tabular file: ✅")
        else:
            info.append("Tabular file: ❌")

        return "\n".join(info)
    except Exception as exc:
        return f"Error getting file info: {exc}"


@tool
def search_files_by_pattern(
    pattern: str,
    directory_path: str = os.getcwd(),
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> str:
    """Search for files matching a pattern in a directory.

    Args:
        pattern: The pattern to search for (supports wildcards like *.csv).
        directory_path: The directory to search in. Defaults to the current working directory.
        max_entries: Maximum number of entries to return. Defaults to 1000.

    Returns:
        Files matching the pattern.

    """
    print(f"    * Tool: search_files_by_pattern | {pattern} in {directory_path}")

    try:
        base_path = Path(directory_path).expanduser().resolve()
    except Exception as exc:
        return f"Invalid directory path: {exc}"

    if not base_path.is_dir():
        return f"Directory not found: {base_path}"

    try:
        from glob import glob

        search_pattern = str(base_path / pattern)
        matches = glob(search_pattern)[:max_entries]

        if not matches:
            return f"No files found matching pattern: {pattern}"

        results = []
        for match in matches:
            path = Path(match)
            if path.is_file():
                size = path.stat().st_size
                results.append(f"📄 {path.name} ({size:,} bytes)")

        return f"Found {len(results)} files matching '{pattern}':\n" + "\n".join(results)
    except Exception as exc:
        return f"Error searching files: {exc}"


def _load_file_safe(file_path: Path, max_rows: int = DEFAULT_MAX_ROWS) -> tuple[pd.DataFrame | None, str | None]:
    """Safely load a file with proper error handling."""
    suffix = file_path.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path, nrows=max_rows)
        elif suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, nrows=max_rows)
        elif suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif suffix == ".json":
            df = pd.read_json(file_path, nrows=max_rows)
        elif suffix == ".tsv":
            df = pd.read_csv(file_path, sep="\t", nrows=max_rows)
        elif suffix == ".pkl":
            if not _pickle_loading_allowed():
                return None, "Pickle loading not allowed (set ALLOW_UNSAFE_PICKLE=1 to enable)"
            df = pd.read_pickle(file_path)  # noqa: S301 - pickle loading is gated by user consent
        else:
            return None, f"Unsupported file type: {suffix}"
    except Exception as exc:
        return None, f"Error reading {file_path.name}: {exc}"
    else:
        return df, None  # type: ignore[return-value]
