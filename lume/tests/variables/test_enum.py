import warnings

import pytest

from lume.variables.enum import EnumVariable


class TestEnumVariable:
    """Test EnumVariable creation and validation."""

    def test_options_validation_empty(self):
        with pytest.raises(ValueError, match="options must not be empty"):
            EnumVariable(name="test", options={})

    def test_options_validation_non_dict(self):
        with pytest.raises(ValueError, match="options must be a dictionary"):
            EnumVariable(name="test", options=["a", "b", "c"])

    def test_options_validation_non_int_keys(self):
        with pytest.raises(ValueError, match="All option keys must be integers"):
            EnumVariable(name="test", options={"a": "label_a", "b": "label_b"})

    def test_options_validation_non_str_values(self):
        with pytest.raises(ValueError, match="All option values must be strings"):
            EnumVariable(name="test", options={0: "red", 1: 2})

    def test_default_value_with_int_key(self):
        var = EnumVariable(
            name="test", options={0: "red", 1: "green", 2: "blue"}, default_value=0
        )
        assert var.default_value == 0

    def test_default_value_with_str_label(self):
        var = EnumVariable(
            name="test",
            options={0: "red", 1: "green", 2: "blue"},
            default_value="red",
        )
        assert var.default_value == "red"

    def test_default_value_invalid_int_key(self):
        with pytest.raises(ValueError, match="not a valid key or label"):
            EnumVariable(
                name="test",
                options={0: "red", 1: "green", 2: "blue"},
                default_value=99,
            )

    def test_default_value_invalid_str_label(self):
        with pytest.raises(ValueError, match="not a valid key or label"):
            EnumVariable(
                name="test",
                options={0: "red", 1: "green", 2: "blue"},
                default_value="yellow",
            )

    def test_validate_with_int_key(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        var.validate_value(0, config="error")
        var.validate_value(1, config="error")
        var.validate_value(2, config="error")

    def test_validate_with_str_label(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        var.validate_value("red", config="error")
        var.validate_value("green", config="error")
        var.validate_value("blue", config="error")

    def test_validate_arbitrary_int_mapping(self):
        var = EnumVariable(name="test", options={5: "red", 10: "green", 15: "blue"})
        var.validate_value(5, config="error")
        var.validate_value(10, config="error")
        var.validate_value(15, config="error")
        var.validate_value("red", config="error")
        var.validate_value("green", config="error")
        var.validate_value("blue", config="error")

    def test_reject_invalid_int_key(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        with pytest.raises(ValueError, match="not a valid key or label"):
            var.validate_value(99, config="error")

    def test_reject_invalid_str_label(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        with pytest.raises(ValueError, match="not a valid key or label"):
            var.validate_value("yellow", config="error")

    def test_validate_rejects_non_int_str_types(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        for invalid in [1.5, None, [], {}]:
            with pytest.raises(TypeError, match="Expected value to be int or str"):
                var.validate_value(invalid, config="error")

    def test_validation_with_config_warn(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            var.validate_value(99, config="warn")
            assert len(w) == 1
            assert "not a valid key or label" in str(w[0].message)

    def test_validation_with_config_none(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        # config="none" should not raise or warn for invalid values
        var.validate_value(99, config="none")

    def test_validation_uses_default_config(self):
        var = EnumVariable(
            name="test",
            options={0: "red", 1: "green", 2: "blue"},
            default_validation_config="warn",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            var.validate_value(99)
            assert len(w) == 1

    def test_get_label(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        assert var.get_label(0) == "red"
        assert var.get_label(1) == "green"
        assert var.get_label(2) == "blue"

    def test_get_label_invalid_key(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        with pytest.raises(KeyError):
            var.get_label(99)

    def test_get_key(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        assert var.get_key("red") == 0
        assert var.get_key("green") == 1
        assert var.get_key("blue") == 2

    def test_get_key_invalid_label(self):
        var = EnumVariable(name="test", options={0: "red", 1: "green", 2: "blue"})
        with pytest.raises(ValueError, match="Label 'yellow' not found"):
            var.get_key("yellow")

    def test_get_key_arbitrary_mapping(self):
        var = EnumVariable(name="test", options={5: "red", 10: "green", 15: "blue"})
        assert var.get_key("red") == 5
        assert var.get_key("green") == 10
        assert var.get_key("blue") == 15
