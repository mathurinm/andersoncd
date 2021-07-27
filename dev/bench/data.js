window.BENCHMARK_DATA = {
  "lastUpdate": 1627395386362,
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
          "id": "e405ae8baf789ac1b6ab347e846ee969728ad21f",
          "message": "WIP ENH: add automated benchmark speed",
          "timestamp": "2021-07-27T12:43:44Z",
          "url": "https://github.com/mathurinm/andersoncd/pull/53/commits/e405ae8baf789ac1b6ab347e846ee969728ad21f"
        },
        "date": 1627395384906,
        "tool": "pytest",
        "benches": [
          {
            "name": "andersoncd/tests/bench_speed.py::test_Lasso[10]",
            "value": 7.763877639650128,
            "unit": "iter/sec",
            "range": "stddev: 0.00259475806700103",
            "extra": "mean: 128.80161775000153 msec\nrounds: 8"
          },
          {
            "name": "andersoncd/tests/bench_speed.py::test_Lasso[50]",
            "value": 2.2032493400528868,
            "unit": "iter/sec",
            "range": "stddev: 0.0012462527553859302",
            "extra": "mean: 453.8750934000035 msec\nrounds: 5"
          },
        ]
      }
    ]
  }
}