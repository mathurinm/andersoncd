window.BENCHMARK_DATA = {
  "lastUpdate": 1627398339387,
  "repoUrl": "https://github.com/mathurinm/andersoncd",
  "entries": {
    "Solver benchmark over commits": [
      {
        "commit": {
          "author": {
            "name": "mathurinm",
            "username": "mathurinm"
          },
          "committer": {
            "name": "mathurinm",
            "username": "mathurinm"
          },
          "id": "bc457e6d7fde7ba26f12bd28b79933ae589452ad",
          "message": "WIP ENH: add automated benchmark speed",
          "timestamp": "2021-07-27T12:43:44Z",
          "url": "https://github.com/mathurinm/andersoncd/pull/53/commits/bc457e6d7fde7ba26f12bd28b79933ae589452ad"
        },
        "date": 1627396147375,
        "tool": "pytest",
        "benches": [
          {
            "name": "andersoncd/tests/bench_speed.py::test_Lasso[10]",
            "value": 3.4736096373929253,
            "unit": "iter/sec",
            "range": "stddev: 0.00561731566853641",
            "extra": "mean: 287.8849681999782 msec\nrounds: 5"
          },
          {
            "name": "andersoncd/tests/bench_speed.py::test_Lasso[50]",
            "value": 1.8960679284142763,
            "unit": "iter/sec",
            "range": "stddev: 0.0031996884178019714",
            "extra": "mean: 527.4072648000129 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "mathurinm",
            "username": "mathurinm"
          },
          "committer": {
            "name": "mathurinm",
            "username": "mathurinm"
          },
          "id": "f4896a79b6db4d1c71d3f1e9eaa2f467497c9a32",
          "message": "WIP ENH: add automated benchmark speed",
          "timestamp": "2021-07-27T12:43:44Z",
          "url": "https://github.com/mathurinm/andersoncd/pull/53/commits/f4896a79b6db4d1c71d3f1e9eaa2f467497c9a32"
        },
        "date": 1627398338420,
        "tool": "pytest",
        "benches": [
          {
            "name": "andersoncd/tests/bench_speed.py::test_Lasso[10]",
            "value": 8.472564013299781,
            "unit": "iter/sec",
            "range": "stddev: 0.00433048175611804",
            "extra": "mean: 118.02802533332921 msec\nrounds: 9"
          },
          {
            "name": "andersoncd/tests/bench_speed.py::test_Lasso[50]",
            "value": 3.0254043121430003,
            "unit": "iter/sec",
            "range": "stddev: 0.003933305410909813",
            "extra": "mean: 330.53433420000147 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}