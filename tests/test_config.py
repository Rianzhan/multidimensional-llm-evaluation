import json
from pathlib import Path

import pytest

from llm_eval.config import load_model_configs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "models.json"


def fake_environment() -> dict[str, str]:
    return {
        "GLM_API_KEY": "glm-test-value",
        "KIMI_API_KEY": "kimi-test-value",
        "DEEPSEEK_API_KEY": "deepseek-test-value",
    }


def test_load_model_configs_reads_keys_only_from_environment():
    configs = load_model_configs(MODEL_CONFIG_PATH, fake_environment())
    by_name = {config.name: config for config in configs}

    assert by_name["GLM"].api_key == "glm-test-value"
    assert by_name["KIMI"].api_key == "kimi-test-value"
    assert by_name["DEEPSEEK"].api_key == "deepseek-test-value"

    raw_config = MODEL_CONFIG_PATH.read_text(encoding="utf-8")
    assert "glm-test-value" not in raw_config
    assert "kimi-test-value" not in raw_config
    assert "deepseek-test-value" not in raw_config
    assert set(item["api_key_env"] for item in json.loads(raw_config)) == {
        "GLM_API_KEY",
        "KIMI_API_KEY",
        "DEEPSEEK_API_KEY",
    }


def test_kimi_has_22_second_interval():
    configs = load_model_configs(MODEL_CONFIG_PATH, fake_environment())
    kimi = next(config for config in configs if config.name == "KIMI")

    assert kimi.min_request_interval_seconds == 22.0


def test_choice_budgets_give_kimi_a_larger_recovery_window():
    configs = load_model_configs(MODEL_CONFIG_PATH, fake_environment())

    for config in configs:
        budget = config.output_budgets_by_task_type["multiple_choice"]
        assert budget.initial == 64
        expected_recovery = 1024 if config.name == "KIMI" else 256
        assert budget.recovery == expected_recovery


def test_free_text_budgets_are_task_specific():
    configs = load_model_configs(MODEL_CONFIG_PATH, fake_environment())

    for config in configs:
        assert config.output_budgets_by_task_type["ifeval"].initial == 2048
        assert config.output_budgets_by_task_type["ifeval"].recovery == 4096
        assert config.output_budgets_by_task_type["product_rubric"].initial == 1024
        assert config.output_budgets_by_task_type["product_rubric"].recovery == 2048


def test_missing_key_names_models_without_echoing_available_secrets():
    environment = {
        "GLM_API_KEY": "available-glm-value",
        "DEEPSEEK_API_KEY": "available-deepseek-value",
    }

    with pytest.raises(ValueError) as exc_info:
        load_model_configs(MODEL_CONFIG_PATH, environment)

    message = str(exc_info.value)
    assert "KIMI" in message
    assert "KIMI_API_KEY" in message
    assert "available-glm-value" not in message
    assert "available-deepseek-value" not in message


def test_delivery_source_and_config_have_no_literal_api_keys():
    candidates = [PROJECT_ROOT / "evaluate.py"]
    candidates.extend((PROJECT_ROOT / "src" / "llm_eval").glob("*.py"))
    candidates.extend((PROJECT_ROOT / "config").glob("*.json"))

    for path in candidates:
        text = path.read_text(encoding="utf-8")
        assert "sk-" not in text, f"literal API key marker found in {path}"
