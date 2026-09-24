"""APC Primitive Bank: structured skill storage, consolidation, and temporary lifecycle."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from .primitives import NAMES, NAMES20
from .runner import json_write, utc_now


@dataclass
class BankEntry:
    id: str
    name: str
    kind: str  # "core", "temporary", "consolidated"
    model_kind: str  # "scripted", "cart", "mlp", "ik_target"
    primitive_count: int
    parameter_count: int
    stored_values: int
    file_bytes: int
    sha256: str
    relative_path: str
    feature_schema: str | None = None
    created_at: str = ""
    consolidated_at: str | None = None
    metadata: dict[str, Any] = None


class PrimitiveBank:
    """Manages versioned physical and neural primitives with explicit temporary release."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.bank_file = self.root / "bank.json"
        self.primitives_dir = self.root / "primitives"
        self.entries: dict[str, BankEntry] = {}
        if self.bank_file.exists():
            self._load()
        else:
            self._init_empty()

    def _init_empty(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.primitives_dir.mkdir(parents=True, exist_ok=True)
        self.save()

    def _load(self):
        data = json.loads(self.bank_file.read_text(encoding="utf-8"))
        self.entries = {k: BankEntry(**v) for k, v in data.get("entries", {}).items()}

    def save(self):
        data = {
            "version": "apc_primitive_bank_v1",
            "updated_at": utc_now(),
            "entry_count": len(self.entries),
            "consolidated_count": sum(1 for e in self.entries.values() if e.kind == "consolidated"),
            "temporary_count": sum(1 for e in self.entries.values() if e.kind == "temporary"),
            "core_count": sum(1 for e in self.entries.values() if e.kind == "core"),
            "entries": {k: asdict(v) for k, v in self.entries.items()}
        }
        json_write(self.bank_file, data)

    def register(self, entry_id: str, name: str, source_file: Path, *,
                 kind: str = "temporary", model_kind: str, primitive_count: int,
                 parameter_count: int, stored_values: int, feature_schema: str | None = None,
                 metadata: dict[str, Any] | None = None) -> BankEntry:
        if kind not in ("core", "temporary", "consolidated"):
            raise ValueError(f"Unknown primitive kind: {kind}")
        dest_filename = f"{entry_id}_{source_file.name}"
        dest_path = self.primitives_dir / dest_filename
        shutil.copy2(source_file, dest_path)
        digest = hashlib.sha256(dest_path.read_bytes()).hexdigest()
        entry = BankEntry(
            id=entry_id,
            name=name,
            kind=kind,
            model_kind=model_kind,
            primitive_count=primitive_count,
            parameter_count=parameter_count,
            stored_values=stored_values,
            file_bytes=dest_path.stat().st_size,
            sha256=digest,
            relative_path=str(dest_path.relative_to(self.root)),
            feature_schema=feature_schema,
            created_at=utc_now(),
            consolidated_at=utc_now() if kind == "consolidated" else None,
            metadata=metadata or {}
        )
        self.entries[entry_id] = entry
        self.save()
        return entry

    def consolidate(self, entry_id: str, candidate_file: Path, *,
                    model_kind: str, parameter_count: int, stored_values: int,
                    metadata: dict[str, Any] | None = None) -> BankEntry:
        """Consolidate a temporary skill into a permanent compact candidate."""
        if entry_id not in self.entries:
            raise KeyError(f"Entry {entry_id} not found in bank")
        entry = self.entries[entry_id]
        if entry.kind != "temporary":
            raise ValueError(f"Entry {entry_id} is already {entry.kind}")
        
        temp_params = entry.parameter_count
        temp_bytes = entry.file_bytes
        
        dest_filename = f"{entry_id}_consolidated_{candidate_file.name}"
        dest_path = self.primitives_dir / dest_filename
        shutil.copy2(candidate_file, dest_path)
        digest = hashlib.sha256(dest_path.read_bytes()).hexdigest()

        entry.kind = "consolidated"
        entry.model_kind = model_kind
        entry.parameter_count = parameter_count
        entry.stored_values = stored_values
        entry.file_bytes = dest_path.stat().st_size
        entry.sha256 = digest
        entry.relative_path = str(dest_path.relative_to(self.root))
        entry.consolidated_at = utc_now()
        meta = entry.metadata or {}
        meta.update(metadata or {})
        meta["consolidation_stats"] = {
            "temporary_parameters": temp_params,
            "consolidated_parameters": parameter_count,
            "parameter_reduction": temp_params - parameter_count,
            "temporary_bytes": temp_bytes,
            "consolidated_bytes": entry.file_bytes,
            "bytes_reduction": temp_bytes - entry.file_bytes
        }
        entry.metadata = meta
        self.save()
        return entry

    def release_temporary(self, entry_id: str) -> dict[str, Any]:
        """Release temporary artifacts while preserving the consolidated entry."""
        if entry_id not in self.entries:
            raise KeyError(f"Entry {entry_id} not in bank")
        entry = self.entries[entry_id]
        if entry.kind != "consolidated":
            raise ValueError("Only consolidated primitives can have their temporary ancestors released")
        
        prefix = f"{entry_id}_"
        released_files = []
        released_bytes = 0
        current_file = self.root / entry.relative_path
        for p in self.primitives_dir.glob(f"{prefix}*"):
            if p.resolve() != current_file.resolve() and "consolidated" not in p.name:
                released_bytes += p.stat().st_size
                released_files.append(str(p.relative_to(self.root)))
                p.unlink()

        release_audit = {
            "entry_id": entry_id,
            "released_at": utc_now(),
            "released_files": released_files,
            "reclaimed_bytes": released_bytes,
            "reclaimed_parameters": entry.metadata.get("consolidation_stats", {}).get("parameter_reduction", 0),
            "status": "fully_released"
        }
        entry.metadata["release_audit"] = release_audit
        self.save()
        return release_audit

    def get_path(self, entry_id: str) -> Path:
        if entry_id not in self.entries:
            raise KeyError(f"Entry {entry_id} not in bank")
        return self.root / self.entries[entry_id].relative_path

    def audit(self) -> dict[str, Any]:
        """Audit physical integrity of all registered primitives."""
        results = {}
        for k, e in self.entries.items():
            path = self.root / e.relative_path
            exists = path.exists()
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if exists else None
            results[k] = {
                "exists": exists,
                "sha256_match": (digest == e.sha256) if exists else False,
                "kind": e.kind,
                "bytes": e.file_bytes,
                "parameters": e.parameter_count
            }
        return {
            "bank_file": str(self.bank_file),
            "entry_count": len(self.entries),
            "audit_timestamp": utc_now(),
            "entries": results
        }
