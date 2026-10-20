window.BENCHMARK_DATA = {
  "lastUpdate": 1792496549236,
  "repoUrl": "https://github.com/yarden-carmi/ipu-emulator",
  "entries": {
    "Times and memory (smaller is better)": [
      {
        "commit": {
          "author": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "committer": {
            "name": "github-actions[bot]",
            "username": "github-actions[bot]",
            "email": "41898282+github-actions[bot]@users.noreply.github.com"
          },
          "id": "0ca490a9ac5a1407e54e7bf561959896c8b08d22",
          "message": "docs: refresh release log",
          "timestamp": "2026-10-19T08:24:25Z",
          "url": "https://github.com/yarden-carmi/ipu-emulator/commit/0ca490a9ac5a1407e54e7bf561959896c8b08d22"
        },
        "date": 1792496548290,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.634017,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.005305,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.685322,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.178267,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.021229,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.058478,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "emulator/ipu_state_rss_delta_mb",
            "value": 0,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "import/ipu_as",
            "value": 0.113448,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.134875,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.228572,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.390625,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 241.652344,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=20.350 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 1888,
            "unit": "s",
            "extra": "ci run 39222493085; whole job 1905 s"
          }
        ]
      }
    ]
  }
}