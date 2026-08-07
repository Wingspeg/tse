"""End-to-end smoke tests that don't touch the GPU or download datasets."""

from __future__ import annotations

import json

import pytest
import torch

from pef.config import D_REPR


class TestRepresentationSmoke:
    def test_generate_returns_descriptor_and_phi(self, small_repr_gen) -> None:
        from pef.representation import ResourceType, extract_algo_attributes

        model_params = {
            "w0": torch.randn(8, 4, dtype=torch.float64),
            "w1": torch.randn(4, 2, dtype=torch.float64),
        }
        attr = extract_algo_attributes(dict(model_params))
        descriptor, phi = small_repr_gen.generate(model_params, ResourceType.ALGO, attr)
        # Descriptor must contain id, attr, com, sigma
        for k in ("id", "attr", "com", "sigma"):
            assert k in descriptor
        assert phi.shape == (D_REPR,)
        assert phi.dtype == torch.float64

    @pytest.mark.parametrize(
        "resource_type, payload_builder",
        [
            ("algo", lambda: {"w": torch.randn(16, 8, dtype=torch.float64)}),
            ("comp", lambda: torch.tensor([10.0, 5.0, 20.0])),
        ],
    )
    def test_all_three_resource_types_round_trip(
        self, small_repr_gen, resource_type, payload_builder
    ) -> None:

        # Only call the two that don't need attributes; the data type
        # gets its own test.
        from pef.representation import extract_algo_attributes, extract_comp_attributes

        if resource_type == "algo":
            payload = payload_builder()
            attr = extract_algo_attributes(dict(payload))
        else:
            payload = payload_builder()
            attr = extract_comp_attributes(10.0, 5.0, 20.0)
        _, phi = small_repr_gen.generate(payload, resource_type, attr)
        assert phi.shape == (D_REPR,)


class TestShapleyFoldingIntegration:
    """The two core mechanisms, wired up as in a real round."""

    def test_shapley_then_folding(self, small_repr_gen, small_lattice_params, tmp_path) -> None:
        from pef.collaborative_folding import CollaborativeFoldingProtocol
        from pef.representation import ResourceType, extract_algo_attributes
        from pef.shapley_enhancement import adaptive_security_enhancement

        # 3 clients, each with 3 resource representations.
        a, c, d = [], [], []
        for i in range(3):
            params = {"w": torch.randn(16, 4, dtype=torch.float64) * (1 + i * 0.2)}
            attr = extract_algo_attributes(dict(params))
            _, pa = small_repr_gen.generate(params, ResourceType.ALGO, attr)
            a.append(pa)
            comp = torch.tensor([10.0, 5.0, 20.0], dtype=torch.float64)
            _, pc = small_repr_gen.generate(comp, ResourceType.COMP, attr)
            c.append(pc)
            data = (f"client_{i}").encode() * 32
            _, pd = small_repr_gen.generate(data, ResourceType.DATA, attr)
            d.append(pd)

        result = adaptive_security_enhancement(a, c, d)
        enhanced = result["enhanced_a"]
        assert len(enhanced) == 3

        # Feed the enhanced representations into the folding protocol.
        proto = CollaborativeFoldingProtocol(small_lattice_params)
        weights = [1 / 3, 1 / 3, 1 / 3]
        for i, (w, p) in enumerate(zip(weights, enhanced, strict=False)):
            proto.publish_commitment(i, p)
            proto.fold_participant(i, w, p)
        assert proto.verify_aggregation()


class TestConfigSmoke:
    def test_config_constants_are_consistent(self) -> None:
        from pef.config import D1, D2, D_REPR, M_LAT, N_LAT, Q

        assert D_REPR == D1 + D2
        assert N_LAT <= M_LAT
        # Q is prime in the framework (12289). Just sanity check it is large.
        assert Q > 2**13


class TestExtrasSmoke:
    def test_run_enhance_returns_expected_keys(self) -> None:
        from pef.experiments_extra import build_real_triples, run_enhance

        a, c, d = build_real_triples(n=4, strength=4.0, seed_offset=99)
        result = run_enhance(a, c, d, max_iters=4)
        for key in ("final_mi", "iters", "noise_energy", "mi", "shapley", "sigma", "enhanced"):
            assert key in result
        # All three lists of enhanced representations have the same length.
        ea, ec, ed = result["enhanced"]
        assert ea.shape[0] == ec.shape[0] == ed.shape[0] == 4


class TestResultsLayout:
    def test_results_dir_writable(self, tmp_path) -> None:
        """The CLI relies on writing JSON files into ./results; smoke it."""
        out = tmp_path / "results"
        out.mkdir()
        path = out / "smoke.json"
        payload = {"ok": True, "n": 3}
        path.write_text(json.dumps(payload))
        loaded = json.loads(path.read_text())
        assert loaded["ok"] is True and loaded["n"] == 3
