window.BENCHMARK_DATA = {
  "lastUpdate": 1627396149220,
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
      }
    ]
  }
}