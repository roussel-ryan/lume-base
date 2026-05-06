import warnings
from pydantic import field_validator, model_validator

from lume.variables.variable import Variable, ConfigEnum


class EnumVariable(Variable):
    """Variable for enumerated values with integer-to-string mapping.

    Similar to EPICS mbbi/mbbo records, this variable allows arbitrary
    integer values to be associated with string labels.

    Attributes
    ----------
    default_value : int | str | None
        Default value for the variable. Can be an integer key or string value.
    options : dict[int, str]
        Mapping of integer values to string labels.
    """

    default_value: int | str | None = None
    options: dict[int, str]

    @field_validator("options", mode="before")
    @classmethod
    def validate_options(cls, value):
        if not value:
            raise ValueError("options must not be empty.")
        if not isinstance(value, dict):
            raise ValueError(
                "options must be a dictionary mapping integers to strings."
            )
        for key, val in value.items():
            if not isinstance(key, int):
                raise ValueError(
                    f"All option keys must be integers, got {type(key).__name__}: {key}."
                )
            if not isinstance(val, str):
                raise ValueError(
                    f"All option values must be strings, got {type(val).__name__}: {val}."
                )
        return value

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            self.validate_value(self.default_value, ConfigEnum.ERROR)
        return self

    def validate_value(self, value: int | str, config: ConfigEnum = None):
        """Validates the given value.

        Parameters
        ----------
        value : int | str
            The value to be validated. Can be an integer key or string label.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error".

        Raises
        ------
        TypeError
            If the value is not an int or str.
        ValueError
            If the value is not a valid key or label in the options mapping.

        """
        # mandatory validation
        self._validate_value_type(value)

        # optional validation
        config = self._validation_config_as_enum(config)

        if config != ConfigEnum.NULL:
            self._validate_value_is_in_options(value, config=config)

    @staticmethod
    def _validate_value_type(value: int | str):
        """Validate that value is int or str."""
        if not isinstance(value, (int, str)):
            raise TypeError(
                f"Expected value to be int or str, but received {type(value).__name__}."
            )

    def _validate_value_is_in_options(
        self, value: int | str, config: ConfigEnum = None
    ):
        """Validate that value is a valid key or label in the options mapping.

        Parameters
        ----------
        value : int | str
            The value to validate.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.

        Raises
        ------
        ValueError
            If the value is not a valid key or label.
        """
        config = self._validation_config_as_enum(config)

        valid_key = isinstance(value, int) and value in self.options
        valid_label = isinstance(value, str) and value in self.options.values()

        if not (valid_key or valid_label):
            error_message = (
                f"Value '{value}' of '{self.name}' is not a valid key or label. "
                f"Valid keys: {list(self.options.keys())}, "
                f"Valid labels: {list(self.options.values())}."
            )
            if config == ConfigEnum.WARN:
                warnings.warn(error_message)
            else:
                raise ValueError(error_message)

    def get_label(self, key: int) -> str:
        """Get the string label for an integer key.

        Parameters
        ----------
        key : int
            The integer key.

        Returns
        -------
        str
            The corresponding string label.

        Raises
        ------
        KeyError
            If the key does not exist in the mapping.
        """
        return self.options[key]

    def get_key(self, label: str) -> int:
        """Get the integer key for a string label.

        Parameters
        ----------
        label : str
            The string label.

        Returns
        -------
        int
            The corresponding integer key.

        Raises
        ------
        ValueError
            If the label does not exist in the mapping.
        """
        for key, val in self.options.items():
            if val == label:
                return key
        raise ValueError(f"Label '{label}' not found in options mapping.")
