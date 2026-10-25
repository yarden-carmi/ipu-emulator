window.BENCHMARK_DATA = {
  "lastUpdate": 1792928515564,
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
      },
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
        "date": 1792583257660,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.844396,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006501,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.909486,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.265305,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.04585,
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
            "value": 0.143106,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.179051,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.294378,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.78125,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 242.109375,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=34.966 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 1888,
            "unit": "s",
            "extra": "ci run 39222493085; whole job 1905 s"
          }
        ]
      },
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
        "date": 1792669383751,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.870176,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006798,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.925591,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.266103,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.046371,
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
            "value": 0.146871,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.181521,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.303007,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.648438,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 242.03125,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=36.245 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 1888,
            "unit": "s",
            "extra": "ci run 39222493085; whole job 1905 s"
          }
        ]
      },
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
        "date": 1792755636062,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.862415,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006531,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.928976,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.263119,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.044821,
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
            "value": 0.144707,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.176929,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.293936,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.027344,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 241.355469,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=35.641 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 1888,
            "unit": "s",
            "extra": "ci run 39222493085; whole job 1905 s"
          }
        ]
      },
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
        "date": 1792841068203,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.638603,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.005328,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.683895,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.177522,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.027876,
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
            "value": 0.111376,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.132059,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.223873,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.464844,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 241.484375,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=20.296 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 1888,
            "unit": "s",
            "extra": "ci run 39222493085; whole job 1905 s"
          }
        ]
      },
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
        "date": 1792928514638,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.883353,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006487,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.927401,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.265978,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.045259,
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
            "value": 0.142842,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.177213,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.292009,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.121094,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 241.363281,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=35.591 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 1888,
            "unit": "s",
            "extra": "ci run 39222493085; whole job 1905 s"
          }
        ]
      }
    ],
    "Rates (bigger is better)": [
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
        "date": 1792496550863,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 13021.726859,
            "unit": "words/s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 8929.883507,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 8.643 s"
          },
          {
            "name": "emulator/faithful/fully_connected.int8/cycles_per_s",
            "value": 16189.170592,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.165 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 15194.294714,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.176 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 10413.827112,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 8809.938606,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=20.350 s"
          }
        ]
      },
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
        "date": 1792583259902,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 9777.400484,
            "unit": "words/s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 5159.055207,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 14.961 s"
          },
          {
            "name": "emulator/faithful/fully_connected.int8/cycles_per_s",
            "value": 9479.121393,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.282 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 8641.794776,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.309 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 7301.966377,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 5127.346201,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=34.966 s"
          }
        ]
      },
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
        "date": 1792669386901,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 9487.738824,
            "unit": "words/s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 5062.148829,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 15.247 s"
          },
          {
            "name": "emulator/faithful/fully_connected.int8/cycles_per_s",
            "value": 9344.558154,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.286 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 8545.42414,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.313 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 7369.41079,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 4946.386169,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=36.245 s"
          }
        ]
      },
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
        "date": 1792755638702,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 9573.120636,
            "unit": "words/s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 5082.186481,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 15.187 s"
          },
          {
            "name": "emulator/faithful/fully_connected.int8/cycles_per_s",
            "value": 9541.619289,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.280 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 8541.649574,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.313 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 7282.51934,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 5030.228408,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=35.641 s"
          }
        ]
      },
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
        "date": 1792841071347,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 12928.221419,
            "unit": "words/s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 8993.621814,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 8.582 s"
          },
          {
            "name": "emulator/faithful/fully_connected.int8/cycles_per_s",
            "value": 16103.713407,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.166 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 15154.696724,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.176 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 11117.869031,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 8833.198309,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=20.296 s"
          }
        ]
      }
    ]
  }
}