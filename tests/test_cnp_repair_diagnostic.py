from __future__ import annotations

from apc.cnp.adaptation import _data_manifest
from apc.cnp.repair_diagnostic import (
    r001_legacy_replay_records,
    r001_new_records,
    r001_old_shadow_records,
)


def test_r001_roles_are_fixed_sized_and_split_disjoint() -> None:
    new_train = r001_new_records("repair_adapt_train")
    new_shadow = r001_new_records("repair_new_shadow")
    old_shadow = r001_old_shadow_records()
    legacy_replay = r001_legacy_replay_records()
    assert {len(panel) for panel in (new_train, new_shadow, old_shadow, legacy_replay)} == {1536}
    manifest = _data_manifest(
        {
            "new_train": new_train,
            "new_shadow": new_shadow,
            "old_shadow": old_shadow,
            "legacy_replay": legacy_replay,
        }
    )
    assert manifest["status"] == "PASS"
