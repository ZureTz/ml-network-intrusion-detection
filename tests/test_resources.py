import time

import psutil

from ml_network_intrusion_detection.common import ResourceMonitor


def test_self_memory_still_recorded_without_child_permissions(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("not permitted")

    monkeypatch.setattr(psutil.Process, "children", denied)
    with ResourceMonitor() as monitor:
        time.sleep(0.06)
    assert monitor.result["peak_rss_bytes"] > 0
    assert monitor.result["memory_sampling_incomplete"]
    assert "child enumeration unavailable" in monitor.result["memory_method"]
