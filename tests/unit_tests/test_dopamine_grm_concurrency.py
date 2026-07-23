from __future__ import annotations

import time

from rlinf.models.embodiment.reward.dopamine_grm_reward_model import DopamineGRMRewardModel


def test_grm_parallel_request_path_preserves_input_order(monkeypatch):
    model = object.__new__(DopamineGRMRewardModel)
    model.endpoint = "http://example.invalid/v1/chat/completions"
    model.model_name = "stackcube-grm-lora"
    model.max_tokens = 16
    model.temperature = 0.0
    model.request_timeout = 1.0
    model.request_workers = 3
    model._build_messages = lambda task, images: [{"role": "user", "content": task}]

    class _Response:
        def __init__(self, index):
            self.index = index

        def read(self):
            return ('{"choices":[{"message":{"content":"%s"}}]}' % self.index).encode()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_urlopen(request, timeout):
        body = request.data.decode()
        index = body.split('"content": "')[1].split('"')[0]
        time.sleep(0.01 * (2 - int(index)))
        return _Response(index)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    outputs, latencies = model._request_grm(
        [{"task": str(index), "images": []} for index in range(3)]
    )
    assert outputs == ["0", "1", "2"]
    assert len(latencies) == 3
