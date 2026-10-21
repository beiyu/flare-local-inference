"""Preparation selects the same IPv4 RoCEv2 rails as native package reload."""

from pathlib import Path

import pytest

from ignition.preparation.collectives import discover_roce_devices

V4 = "0000:0000:0000:0000:0000:ffff:0a4d:0d01"
V6 = "fe80:0000:0000:0000:4ebb:47ff:fe7f:0431"


def device(root, name, gids, *, state="4: ACTIVE", port=1):
    directory = root / name / f"ports/{port}"
    directory.mkdir(parents=True)
    (directory / "state").write_text(state)
    for index, (kind, gid) in gids.items():
        for part, value in (("gid_attrs/types", kind), ("gids", gid)):
            path = directory / part / str(index)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)


def test_active_link_local_rail_is_excluded_from_ipv4_fabric(tmp_path):
    device(tmp_path, "rocep1s0f0", {0: ("RoCE v1", V6), 1: ("RoCE v2", V6),
                                  2: ("RoCE v1", V4), 3: ("RoCE v2", V4)})
    device(tmp_path, "roceP2p1s0f0", {1: ("RoCE v2", V6), 3: ("RoCE v2", V6)})
    device(tmp_path, "rocep1s0f1", {3: ("RoCE v2", V4)}, state="1: DOWN")
    assert discover_roce_devices(tmp_path) == (["rocep1s0f0"], 3)


def test_preferred_index_three_wins_when_every_active_device_has_it(tmp_path):
    device(tmp_path, "b", {1: ("RoCE v2", V4), 3: ("RoCE v2", V4)})
    device(tmp_path, "a", {1: ("RoCE v2", V4), 3: ("roce V2\n", V4)})
    assert discover_roce_devices(tmp_path) == (["a", "b"], 3)


def test_fallback_uses_lowest_any_eligible_index_and_only_its_devices(tmp_path):
    device(tmp_path, "b", {5: ("RoCE v2", V4)})
    device(tmp_path, "a", {2: ("RoCE v2", V4), 3: ("RoCE v2", V4)})
    device(tmp_path, "v1", {0: ("RoCE v1", V4)})
    assert discover_roce_devices(tmp_path) == (["a"], 2)


def test_eligible_devices_are_sorted_before_the_four_rail_limit(tmp_path):
    for name in ("g", "c", "e", "a", "f", "b", "d"):
        device(tmp_path, name, {3: ("RoCE v2", V4)})
    assert discover_roce_devices(tmp_path) == (["a", "b", "c", "d"], 3)


@pytest.mark.parametrize("kind,gid", [("RoCE v1", V4), ("RoCE v2", V6),
    ("RoCE v2", "0000:0000:0000:0000:0000:0000:0000:0000"),
    ("RoCE v2", "0000:0000:0000:0000:0000:ffff:zzzz:0001")])
def test_non_ipv4_or_v1_gid_is_rejected(tmp_path, kind, gid):
    device(tmp_path, "a", {3: (kind, gid)})
    with pytest.raises(RuntimeError, match="no active RDMA device has a RoCEv2 IPv4 GID"):
        discover_roce_devices(tmp_path)


def test_only_port_one_and_indices_zero_through_fifteen_are_candidates(tmp_path):
    device(tmp_path, "a", {3: ("RoCE v2", V4)}, port=2)
    device(tmp_path, "b", {16: ("RoCE v2", V4)})
    with pytest.raises(RuntimeError, match="no active RDMA device"):
        discover_roce_devices(tmp_path)


def test_missing_or_empty_sysfs_root_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="cannot list RDMA devices"):
        discover_roce_devices(tmp_path / "missing")
    with pytest.raises(RuntimeError, match="no active RDMA device"):
        discover_roce_devices(tmp_path)
