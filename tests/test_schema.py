import pandas as pd
import pytest

from gpufleet.schema import COLUMN_NAMES, SchemaError, schema_markdown, validate_telemetry


def test_simulated_telemetry_is_valid(sim):
    validate_telemetry(sim.telemetry)
    assert list(sim.telemetry.columns) == COLUMN_NAMES


def test_missing_column_is_reported(sim):
    with pytest.raises(SchemaError, match="missing columns"):
        validate_telemetry(sim.telemetry.drop(columns=["power_w"]))


def test_out_of_range_value_is_reported(sim):
    bad = sim.telemetry.head(100).copy()
    bad.loc[bad.index[0], "gpu_util"] = 150.0
    with pytest.raises(SchemaError, match="gpu_util: values above 100"):
        validate_telemetry(bad)


def test_decreasing_counter_is_reported(sim):
    tel = sim.telemetry
    uuid = tel.loc[tel["ecc_sbe_total"] > 0, "gpu_uuid"].iloc[0]
    bad = tel[tel["gpu_uuid"] == uuid].copy()
    bad.loc[bad.index[-1], "ecc_sbe_total"] = 0
    with pytest.raises(SchemaError, match="ecc_sbe_total: counter decreases"):
        validate_telemetry(bad)


def test_duplicate_rows_are_reported(sim):
    head = sim.telemetry.head(10)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_telemetry(pd.concat([head, head]))


def test_schema_markdown_lists_every_column():
    md = schema_markdown()
    for name in COLUMN_NAMES:
        assert f"`{name}`" in md
