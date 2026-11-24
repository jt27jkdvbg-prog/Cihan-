
import json

import pytest

from wan_performance import PerformanceRecorder


def test_stage_records_success(tmp_path):
    recorder = PerformanceRecorder(tmp_path)

    with recorder.stage("model_load"):
        pass

    report = recorder.build_report()

    assert report["stage_count"] == 1
    assert report["successful_stage_count"] == 1
    assert report["failed_stage_count"] == 0
    assert report["stages"][0]["name"] == "model_load"
    assert report["stages"][0]["status"] == "success"


def test_stage_records_failure_and_reraises(tmp_path):
    recorder = PerformanceRecorder(tmp_path)

    with pytest.raises(RuntimeError):
        with recorder.stage("video_generation"):
            raise RuntimeError("test failure")

    report = recorder.build_report()

    assert report["failed_stage_count"] == 1
    assert report["stages"][0]["status"] == "failed"
    assert report["stages"][0]["error_type"] == "RuntimeError"


def test_scene_metrics_and_average(tmp_path):
    recorder = PerformanceRecorder(tmp_path)

    recorder.record_scene(1, 10.0, 81, 832, 480, 8, seed=123)
    recorder.record_scene(2, 20.0, 81, 832, 480, 8, seed=456)

    report = recorder.build_report()

    assert report["scene_count"] == 2
    assert report["average_scene_seconds"] == 15.0
    assert report["scenes"][0]["seed"] == 123


def test_negative_scene_metric_is_rejected(tmp_path):
    recorder = PerformanceRecorder(tmp_path)

    with pytest.raises(ValueError):
        recorder.record_scene(1, -1, 81, 832, 480, 8)


def test_report_is_written_as_valid_json(tmp_path):
    recorder = PerformanceRecorder(
        tmp_path,
        project_title="Performance Test",
        profile="landscape",
    )

    path = recorder.write_report()

    assert path.is_file()

    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["project_title"] == "Performance Test"
    assert report["profile"] == "landscape"
    assert report["report_version"] == 1


def test_gpu_capture_handles_unavailable_cuda(tmp_path):
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    recorder = PerformanceRecorder(tmp_path)
    sample = recorder.capture_gpu(FakeTorch)

    assert sample["available"] is False
    assert len(recorder.gpu_samples) == 1
