from pathlib import Path


def test_plug_awbc_sft_config_defaults_to_one_explicit_gpu_rank():
    config = (
        Path(__file__).parents[2] / "examples" / "sft" / "config" / "maniskill_plug_awbc_sft_openpi_pi05.yaml"
    ).read_text(encoding="utf-8")
    assert "actor,env,rollout: ${oc.env:ASK4HELP_RLINF_PLACEMENT,0-0}" in config
    assert "actor,env,rollout: all" not in config
