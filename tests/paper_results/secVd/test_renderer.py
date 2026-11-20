"""Viser-renderer regressions for Section Vd."""

import sys
from pathlib import Path

import jax
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

SECTION_DIR = (
    Path(__file__).resolve().parents[3]
    / "paper_results"
    / "secVd_control_gain_optimization"
)
CODE_DIR = SECTION_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import render_control_gain_optimization_animations as renderer  # noqa: E402
from secvd_case import build_sec_vd_robot  # noqa: E402
from secvd_results import load_results  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_resampling_matches_requested_fps():
    t = np.linspace(0, 1, 101)
    q = np.column_stack([t, 2 * t])
    pose = np.column_stack([t] * 6)
    frame_t, frame_q, frame_pose = renderer.resample_trajectory(t, q, pose, fps=10)
    assert len(frame_t) == 11
    np.testing.assert_allclose(frame_q[:, 0], frame_t)
    np.testing.assert_allclose(frame_pose[:, 5], frame_t)


def test_desired_shape_wireframe_is_finite_and_three_dimensional():
    z = np.linspace(0.0, 0.1, 12)
    curve = np.column_stack([0.01 * np.sin(8 * z), np.zeros_like(z), z])
    segments = renderer.make_tube_wire_segments(curve, radius=0.02)
    assert segments.ndim == 3
    assert segments.shape[1:] == (2, 3)
    assert np.all(np.isfinite(segments))


def test_method_selection_and_overwrite(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        renderer,
        "render_method",
        lambda **kwargs: (
            calls.append(kwargs["name"]) or [tmp_path / f"{kwargs['name']}.mp4"]
        ),
    )
    outputs = renderer.render_saved_results(
        data_dir=tmp_path,
        output_dir=tmp_path,
        method="synergistic",
        renderer_factory=object,
    )
    assert calls == ["synergistic"]
    assert outputs == [tmp_path / "synergistic.mp4"]

    (tmp_path / "collocated_optimized_best_tracking.mp4").touch()
    with pytest.raises(FileExistsError, match="--force"):
        renderer.render_saved_results(
            data_dir=tmp_path,
            output_dir=tmp_path,
            method="both",
            renderer_factory=object,
        )
    assert calls == ["synergistic"], "preflight must fail before either render starts"

    (tmp_path / "collocated_optimized_best_tracking.mp4").unlink()
    (tmp_path / "collocated_initial_median_tracking.mp4").touch()
    with pytest.raises(FileExistsError, match="--force"):
        renderer.render_saved_results(
            data_dir=tmp_path,
            output_dir=tmp_path,
            method="collocated",
            trajectory="initial-median",
            renderer_factory=object,
        )


def test_mock_viser_receives_robot_trail_target_and_paths(monkeypatch, tmp_path):
    timesteps = 5
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "q_ts_best": np.zeros((1, timesteps, 6)),
        "x_ts_best": np.zeros((1, timesteps, 6)),
        "best_batch": np.asarray(0),
        "x_des_ts": np.zeros((timesteps, 6)),
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, robot, **kwargs):
            captured["init"] = (robot, kwargs)

        def render_sequence(self, **kwargs):
            captured["render"] = kwargs

    outputs = renderer.render_method(
        name="synergistic",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        renderer_factory=FakeRenderer,
    )
    assert outputs == [tmp_path / "synergistic_optimized_best_tracking.mp4"]
    assert captured["render"]["q_ts"].shape == (5, 6)
    assert captured["render"]["camera_config"] == renderer.make_camera_config(0.1)
    assert captured["init"][1]["desired_q_ts"] is None
    assert captured["init"][1]["config"].geometry.cross_section_resolution == 64
    assert "cylinder_sections" not in captured["init"][1]
    backbone_colors = captured["init"][1]["config"].colors.backbone.point_colors
    np.testing.assert_allclose(
        backbone_colors[:, 3], renderer.SYNERGISTIC_ROBOT_OPACITY
    )
    assert captured["render"]["dynamic_spheres_positions"].shape == (2, 5, 3)
    np.testing.assert_allclose(
        captured["render"]["dynamic_spheres_radii"],
        [renderer.MARKER_RADIUS, renderer.TARGET_MARKER_RADIUS],
    )
    np.testing.assert_allclose(
        captured["render"]["dynamic_spheres_colors"],
        np.stack([renderer.CURRENT_POSITION_COLOR, renderer.TARGET_COLOR]),
    )
    assert captured["render"]["static_spheres_positions"] is None
    assert captured["render"]["record_path"].endswith(
        "synergistic_optimized_best_tracking.mp4"
    )

    outputs[0].touch()
    with pytest.raises(FileExistsError, match="--force"):
        renderer.render_method(
            name="synergistic",
            data_dir=tmp_path,
            output_dir=tmp_path,
            fps=4,
            make_gif=False,
            force=False,
            port=8080,
            open_browser=False,
            record_client_timeout=2.0,
            record_frame_timeout=3.0,
            renderer_factory=FakeRenderer,
        )


def test_initial_median_renders_the_representative_initial_rollout(
    monkeypatch, tmp_path
):
    timesteps = 5
    q_init = np.stack(
        [np.full((timesteps, 6), value, dtype=float) for value in (1.0, 2.0, 3.0)]
    )
    x_init = np.stack(
        [np.full((timesteps, 6), value, dtype=float) for value in (4.0, 5.0, 6.0)]
    )
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "history_loss": np.array([[9.0, 2.0, 5.0]]),
        "history_finite_mask": np.ones((1, 3), dtype=bool),
        "q_ts_init": q_init,
        "x_ts_init": x_init,
        "q_ts_best": np.zeros_like(q_init),
        "x_ts_best": np.zeros_like(x_init),
        "best_batch": np.asarray(0),
        "x_des_ts": np.zeros((timesteps, 6)),
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, _robot, **kwargs):
            captured["init"] = kwargs

        def render_sequence(self, **kwargs):
            captured["render"] = kwargs

    outputs = renderer.render_method(
        name="synergistic",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        trajectory="initial-median",
        renderer_factory=FakeRenderer,
    )

    assert outputs == [tmp_path / "synergistic_initial_median_tracking.mp4"]
    np.testing.assert_allclose(captured["render"]["q_ts"], q_init[2])
    assert captured["render"]["camera_config"] == renderer.make_camera_config(0.1)
    assert captured["render"]["robot_name"].endswith("initial median")


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        ([7.0, 5.0, 9.0, 3.0], 1),
        ([5.0, 5.0, 9.0], 0),
    ],
)
def test_initial_median_ties_prefer_lower_loss_then_batch_index(loss, expected):
    data = {
        "history_loss": np.array([loss]),
        "history_finite_mask": np.ones((1, len(loss)), dtype=bool),
    }
    assert renderer.initial_median_batch(data) == expected


def test_initial_median_requires_a_finite_initial_loss():
    data = {
        "history_loss": np.array([[np.nan, 2.0]]),
        "history_finite_mask": np.array([[False, False]]),
    }
    with pytest.raises(ValueError, match="finite initial loss"):
        renderer.initial_median_batch(data)


def test_shared_camera_is_centered_framed_and_gravity_down():
    total_length = 0.1
    camera = renderer.make_camera_config(total_length)
    position = np.asarray(camera.position)
    look_at = np.asarray(camera.look_at)
    view_offset = position - look_at
    vertical_span = (
        2.0 * np.linalg.norm(view_offset) * np.tan(np.deg2rad(camera.fov / 2.0))
    )

    assert camera.up == (0.0, 0.0, -1.0)
    assert position[0] == look_at[0] == 0.0
    assert view_offset[2] > 0.0  # Reveal the overhead mounting surface.
    assert look_at[2] == pytest.approx(0.25 * total_length)
    assert 2.4 * total_length < vertical_span < 3.0 * total_length


def test_recording_client_flushes_camera_before_the_first_frame(monkeypatch):
    class FakeClient:
        flushed = False

        def flush(self):
            self.flushed = True

    client = FakeClient()
    monkeypatch.setattr(
        renderer.ViserRenderer,
        "_wait_for_recording_client",
        lambda _self, _timeout: client,
    )
    instance = object.__new__(renderer.SecVdTrackingRenderer)

    assert instance._wait_for_recording_client(1.0) is client
    assert client.flushed


def test_first_recorded_frame_follows_a_discarded_camera_warmup(monkeypatch):
    client = object()
    calls = []

    def capture(_self, active_client, *, timeout):
        calls.append((active_client, timeout))
        return np.zeros((1, 1, 3), dtype=np.uint8), client

    monkeypatch.setattr(renderer.ViserRenderer, "_capture_viser_frame", capture)
    instance = object.__new__(renderer.SecVdTrackingRenderer)
    instance._warmed_recording_client = None

    instance._capture_viser_frame(client, timeout=2.0)
    assert calls == [(client, 2.0), (client, 2.0)]
    instance._capture_viser_frame(client, timeout=2.0)
    assert calls == [(client, 2.0), (client, 2.0), (client, 2.0)]


def test_synergistic_trail_contains_only_time_varying_reference(monkeypatch, tmp_path):
    timesteps = 5
    reference_pose = np.zeros((timesteps, 6))
    reference_pose[:, 3] = np.linspace(0.0, 0.04, timesteps)
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "q_ts_best": np.zeros((1, timesteps, 6)),
        "x_ts_best": np.zeros((1, timesteps, 6)),
        "best_batch": np.asarray(0),
        "x_des_ts": reference_pose,
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, _robot, **_kwargs):
            pass

        def render_sequence(self, **kwargs):
            captured.update(kwargs)

    renderer.render_method(
        name="synergistic",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        renderer_factory=FakeRenderer,
    )
    expected_reference = reference_pose[:, 3:6]
    np.testing.assert_allclose(captured["static_spheres_positions"], expected_reference)
    np.testing.assert_allclose(
        captured["static_spheres_colors"],
        np.tile(renderer.TARGET_TRAIL_COLOR, (timesteps, 1)),
    )


def test_collocated_render_receives_configuration_target(monkeypatch, tmp_path):
    timesteps = 5
    data = {
        "t_ts": np.linspace(0, 1, timesteps),
        "q_ts_best": np.zeros((1, timesteps, 6)),
        "q_des_ts": np.ones((timesteps, 6)),
        "x_ts_best": np.zeros((1, timesteps, 6)),
        "best_batch": np.asarray(0),
        "x_des_ts": np.zeros((timesteps, 6)),
    }
    monkeypatch.setattr(renderer, "load_results", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(renderer, "build_sec_vd_robot", lambda: (object(), None, 0.1))
    monkeypatch.setattr(renderer, "validate_pose_consistency", lambda *_args: None)
    captured = {}

    class FakeRenderer:
        def __init__(self, _robot, **kwargs):
            captured["init"] = kwargs

        def render_sequence(self, **kwargs):
            captured["render"] = kwargs

    renderer.render_method(
        name="collocated",
        data_dir=tmp_path,
        output_dir=tmp_path,
        fps=4,
        make_gif=False,
        force=False,
        port=8080,
        open_browser=False,
        record_client_timeout=2.0,
        record_frame_timeout=3.0,
        renderer_factory=FakeRenderer,
    )
    assert captured["init"]["desired_q_ts"].shape == (5, 6)
    np.testing.assert_allclose(
        captured["init"]["config"].colors.backbone.point_colors[:, 3], 1.0
    )
    assert captured["render"]["static_spheres_positions"] is None
    assert captured["render"]["dynamic_spheres_positions"] is None


@pytest.mark.parametrize("method", ["collocated", "synergistic"])
@pytest.mark.parametrize("trajectory", renderer.TRAJECTORIES)
def test_committed_pose_matches_forward_kinematics(method, trajectory):
    data = load_results(SECTION_DIR / "data" / method, expected_method=method)
    robot, _routing, total_length = build_sec_vd_robot()
    renderer.validate_pose_consistency(robot, total_length, data, trajectory)
