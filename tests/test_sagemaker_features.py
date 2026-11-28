"""Tests of the SageMaker Feature Store adapter: names, definitions, records and putting them."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from eventgpt.feature_engineering import HeadFeatureStore, feature_definitions, put_records, records, sagemaker_names


class FakeRuntime:
    """A featurestore-runtime client keeping records as AWS does: online the latest event time per id, offline all."""

    def __init__(self) -> None:
        """Empty stores."""
        self.online: dict[str, dict[str, str]] = {}
        self.offline: list[dict[str, str]] = []
        self.lock = threading.Lock()

    def put_record(self, FeatureGroupName: str, Record: list[dict[str, str]]) -> None:  # noqa: N803  boto3's names
        """Keep the record offline, and online when it is the id's latest."""
        rec = {f["FeatureName"]: f["ValueAsString"] for f in Record}
        with self.lock:
            self.offline.append(rec)
            old = self.online.get(rec["client_id"])
            if old is None or old["event_time"] <= rec["event_time"]:
                self.online[rec["client_id"]] = rec


def table() -> pd.DataFrame:
    """Three cuts of two clients, out of time order, with a missing value."""
    return pd.DataFrame(
        {
            "client_id": ["a", "b", "a", "a"],
            "event_timestamp": pd.to_datetime(["2024-03-31", "2024-01-31", "2024-01-31", "2024-02-29"]),
            "L1H1_s11=22": np.float32([3.0, 7.0, 1.0, np.nan]),
            "L1H1_name=trx_40_count": np.float32([0.5, 0.25, 0.125, 2.0]),
        }
    )


# -------------------------------------------------------------------------------
# Names and definitions
# -------------------------------------------------------------------------------
def test_sagemaker_names_follow_the_feature_name_rules() -> None:
    """Names become lower-case SageMaker names; long, reserved and clashing names are refused."""
    assert sagemaker_names(["L1H1_s11=22", "amt_bin=", "3d"]) == ["l1h1_s11_22", "amt_bin", "3d"]
    with pytest.raises(ValueError, match="at most 64"):
        sagemaker_names(["x" * 65])
    with pytest.raises(ValueError, match="reserved"):
        sagemaker_names(["write_time"])
    with pytest.raises(ValueError, match="both become"):
        sagemaker_names(["a=b", "a_b"])


def test_definitions_type_features_and_put_the_keys_first(tmp_path: Path) -> None:
    """The record id and event time come first as String; float features are Fractional, integers Integral."""
    store = HeadFeatureStore(tmp_path)
    store.add([1], pd.DataFrame({"L1H1_late": np.float32([1.0]), "n_events": np.int64([3])}))
    store.flush()
    defs = feature_definitions(store, "client_id", "event_time")
    assert defs == [
        {"FeatureName": "client_id", "FeatureType": "String"},
        {"FeatureName": "event_time", "FeatureType": "String"},
        {"FeatureName": "l1h1_late", "FeatureType": "Fractional"},
        {"FeatureName": "n_events", "FeatureType": "Integral"},
    ]
    with pytest.raises(AssertionError, match="2500"):
        feature_definitions(store, "client_id", "event_time", ["L1H1_late"] * 2499)


# -------------------------------------------------------------------------------
# Records
# -------------------------------------------------------------------------------
def test_records_are_in_time_order_with_iso_times_and_no_missing_values() -> None:
    """Records come oldest first with UTC ISO event times; a missing value is left out; floats round-trip."""
    recs = list(records(table(), "client_id", "event_timestamp"))
    times = [r[1]["ValueAsString"] for r in recs]
    assert times == sorted(times) and times[0] == "2024-01-31T00:00:00Z"
    feb = next(r for r in recs if r[1]["ValueAsString"].startswith("2024-02"))
    assert [f["FeatureName"] for f in feb] == ["client_id", "event_time", "l1h1_name_trx_40_count"]
    assert all(np.float32(float(f["ValueAsString"])) == np.float32(v) for f, v in zip(recs[-1][2:], [3.0, 0.5], strict=True))


@pytest.mark.parametrize("workers", [1, 4])
def test_put_records_leaves_each_latest_state_online(workers: int) -> None:
    """Online each client has its latest cut, also from several threads; offline every cut."""
    rt: Any = FakeRuntime()
    big = pd.concat([table().assign(client_id=lambda t, k=k: t["client_id"] + str(k)) for k in range(25)], ignore_index=True)
    assert put_records(rt, "heads-0", big, "client_id", "event_timestamp", workers=workers) == 100
    assert len(rt.offline) == 100
    assert {k: v["event_time"][:7] for k, v in rt.online.items() if k in ("a0", "b0")} == {"a0": "2024-03", "b0": "2024-01"}
    assert rt.online["a7"]["l1h1_s11_22"] == "3.0"


def test_last_write_stores_agree_because_records_come_in_time_order() -> None:
    """A store keeping the last write (as oblako's does) ends with the latest state too."""

    class LastWrite(FakeRuntime):
        """Keeps the last record written per id, whatever its event time."""

        def put_record(self, FeatureGroupName: str, Record: list[dict[str, str]]) -> None:  # noqa: N803
            """Overwrite the id's record."""
            rec = {f["FeatureName"]: f["ValueAsString"] for f in Record}
            self.online[rec["client_id"]] = rec

    rt: Any = LastWrite()
    put_records(rt, "heads-0", table(), "client_id", "event_timestamp")
    assert rt.online["a"]["event_time"].startswith("2024-03")
