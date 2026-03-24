from pydantic import field_validator, model_validator
import numpy as np
import warnings

from lume.variables.variable import Variable, ConfigEnum


class StrVariable(Variable):
    """Variable for str values.

    Attributes
    ----------
    default_value : str | None
        Default value for the variable.
    """

    default_value: str | None = None

    @model_validator(mode="after")
    def validate_default_value(self):
        if self.default_value is not None:
            self.validate_value(self.default_value, ConfigEnum.ERROR)
        return self

    def validate_value(self, value: float, config: ConfigEnum = None):
        """Validates the given value.

        Parameters
        ----------
        value : float
            The value to be validated.
        config : ConfigEnum, optional
            The configuration for validation. Defaults to None.
            Allowed values are "none", "warn", and "error".

        Raises
        ------
        TypeError
            If the value is not of type float.
        ValueError
            If the value is out of the valid range or does not match the default value
            for constant variables.

        """
        # mandatory validation
        self._validate_value_type(value)

        # optional validation
        config = self._validation_config_as_enum(config)

        if config != ConfigEnum.NULL:
            self._validate_value_is_within_range(value, config=config)

    @staticmethod
    def _validate_value_type(value: float):
        if not isinstance(value, (int, float, np.integer)):
            raise TypeError(
                f"Expected value to be of type {int} or {np.integer}, but received {type(value)}."
            )

    def _validate_value_is_within_range(self, value: float, config: ConfigEnum = None):
        config = self._validation_config_as_enum(config)
        # Additional validation here
