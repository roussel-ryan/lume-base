from typing import Any

import numpy as np
import pytest
from pmd_beamphysics import ParticleGroup

from lume.model import LUMEModel
from lume.staged_model import StagedModel
from lume.variables import ParticleGroupVariable, ScalarVariable, Variable


def make_test_particle_group(
    n_particles: int = 8,
    x_offset: float = 0.0,
    pz: float = 1.0e7,
) -> ParticleGroup:
    """Build a small deterministic ParticleGroup for staged-model tests."""

    x = np.linspace(-1.0e-3, 1.0e-3, n_particles) + x_offset
    zeros = np.zeros(n_particles)

    return ParticleGroup(
        data={
            "x": x,
            "px": zeros,
            "y": zeros,
            "py": zeros,
            "z": zeros,
            "pz": np.full(n_particles, pz),
            "t": zeros,
            "status": np.ones(n_particles),
            "weight": np.full(n_particles, 1.0e-12),
            "species": "electron",
        }
    )


class BeamSourceTestModel(LUMEModel):
    """Simple source-like model exposing staged beam input/output variables."""

    def __init__(self):
        self.get_call_count = 0
        self.set_call_count = 0

        self._variables = {
            "source_phase": ScalarVariable(
                name="source_phase", default_value=0.0, read_only=False
            ),
            "beam_input": ParticleGroupVariable(name="beam_input", read_only=False),
            "beam_output": ParticleGroupVariable(name="beam_output", read_only=True),
        }

        initial_beam = make_test_particle_group(x_offset=0.0)
        self._state = {
            "source_phase": 0.0,
            "beam_input": initial_beam,
            "beam_output": initial_beam,
        }
        self._initial_state = self._state.copy()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def get(self, names: list[str]) -> dict[str, Any]:
        self.get_call_count += 1
        return super().get(names)

    def set(self, values: dict[str, Any]) -> None:
        self.set_call_count += 1
        super().set(values)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        if "source_phase" in values:
            self._state["source_phase"] = values["source_phase"]

        if "beam_input" in values:
            self._state["beam_input"] = values["beam_input"]

        # Keep behavior deterministic: this source model forwards beam_input to beam_output.
        self._state["beam_output"] = self._state["beam_input"]

    def reset(self) -> None:
        self._state = self._initial_state.copy()


class BeamTransportTestModel(LUMEModel):
    """Simple transport-like model exposing staged beam input/output variables."""

    def __init__(self):
        self.get_call_count = 0
        self.set_call_count = 0

        self._variables = {
            "transport_scale": ScalarVariable(
                name="transport_scale", default_value=1.0, read_only=False
            ),
            "beam_input": ParticleGroupVariable(name="beam_input", read_only=False),
            "beam_output": ParticleGroupVariable(name="beam_output", read_only=True),
        }

        initial_beam = make_test_particle_group(x_offset=2.0e-4)
        self._state = {
            "transport_scale": 1.0,
            "beam_input": initial_beam,
            "beam_output": initial_beam,
        }
        self._initial_state = self._state.copy()

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def get(self, names: list[str]) -> dict[str, Any]:
        self.get_call_count += 1
        return super().get(names)

    def set(self, values: dict[str, Any]) -> None:
        self.set_call_count += 1
        super().set(values)

    def _get(self, names: list[str]) -> dict[str, Any]:
        return {name: self._state[name] for name in names}

    def _set(self, values: dict[str, Any]) -> None:
        if "transport_scale" in values:
            self._state["transport_scale"] = values["transport_scale"]

        if "beam_input" in values:
            self._state["beam_input"] = values["beam_input"]

        # Keep behavior deterministic: this transport model forwards beam_input to beam_output.
        self._state["beam_output"] = self._state["beam_input"]

    def reset(self) -> None:
        self._state = self._initial_state.copy()


def test_staged_model_supported_variables_union() -> None:
    model = StagedModel([BeamSourceTestModel(), BeamTransportTestModel()])

    variable_names = set(model.supported_variables.keys())
    assert {"source_phase", "transport_scale", "beam_input", "beam_output"}.issubset(
        variable_names
    )


def test_staged_model_propagates_beam_to_next_stage() -> None:
    beam_source = BeamSourceTestModel()
    beam_transport = BeamTransportTestModel()
    model = StagedModel([beam_source, beam_transport])

    new_beam = make_test_particle_group(x_offset=7.5e-4)
    model.set({"beam_input": new_beam, "source_phase": 5.0})

    source_beam_out = beam_source.get(["beam_output"])["beam_output"]
    transport_beam_in = beam_transport.get(["beam_input"])["beam_input"]
    transport_beam_out = beam_transport.get(["beam_output"])["beam_output"]

    assert np.allclose(source_beam_out.x, new_beam.x)
    assert np.allclose(transport_beam_in.x, new_beam.x)
    assert np.allclose(transport_beam_out.x, new_beam.x)
    assert beam_source.set_call_count == 1
    assert beam_transport.set_call_count == 1
    assert beam_source.get_call_count == 2
    assert beam_transport.get_call_count == 3


def test_staged_model_only_updates_later_stage_when_requested() -> None:
    beam_source = BeamSourceTestModel()
    beam_transport = BeamTransportTestModel()
    model = StagedModel([beam_source, beam_transport])

    initial_transport_beam_in = beam_transport.get(["beam_input"])["beam_input"]
    initial_transport_beam_in_x = initial_transport_beam_in.x.copy()

    model.set({"transport_scale": 2.5})

    assert beam_transport.get(["transport_scale"])["transport_scale"] == 2.5
    assert np.allclose(
        beam_transport.get(["beam_input"])["beam_input"].x,
        initial_transport_beam_in_x,
    )
    assert beam_source.set_call_count == 0
    assert beam_source.get_call_count == 0
    assert beam_transport.set_call_count == 1
    assert beam_transport.get_call_count == 4


def test_staged_model_requires_beam_output_for_non_last_stage() -> None:
    class MissingBeamOutputModel(BeamSourceTestModel):
        @property
        def supported_variables(self) -> dict[str, Variable]:
            variables = dict(self._variables)
            variables.pop("beam_output")
            return variables

    with pytest.raises(ValueError, match="must have a 'beam_output' variable"):
        StagedModel([MissingBeamOutputModel(), BeamTransportTestModel()])


def test_staged_model_requires_beam_input_after_first_stage() -> None:
    class MissingBeamInputModel(BeamTransportTestModel):
        @property
        def supported_variables(self) -> dict[str, Variable]:
            variables = dict(self._variables)
            variables.pop("beam_input")
            return variables

    with pytest.raises(ValueError, match="must have a 'beam_input' variable"):
        StagedModel([BeamSourceTestModel(), MissingBeamInputModel()])
