# T-Shape MJWarp — proyek terpisah

Backend eksperimen Medium satu pintu dengan simulasi paralel MJWarp dan MAPPO
PyTorch. Proyek ini mandiri: tidak mengimpor, mengedit, atau memasang dependensi
ke `tshape-mappo-mujoco`. Folder `tshape_mjwarp/reference/` adalah salinan referensi
CPU dan definisi jaringan dari baseline; `SOURCE.json` mencatat hash sumbernya.
Resolver lokasi aset pada salinan disesuaikan ke root proyek ini.

## Status verifikasi

Tes logika CPU, GAE, observasi, randomisasi, partial reset, checkpoint/resume,
evaluasi deterministik, benchmark dan video disediakan. Tes CUDA harus dijalankan
di PC remote. Tidak ada angka percepatan GPU yang diasumsikan atau dijanjikan.
Backend `reference` adalah oracle C MuJoCo pada CPU, bukan fallback GPU otomatis.
`mjwarp --device cpu` menjalankan implementasi MJWarp pada CPU untuk debugging;
hasil kecepatannya tidak mewakili GPU.

Tahap pertama mendukung Medium dengan satu pintu, arah melintas +X, objek baseline
dengan COM di origin dan geometri box sejajar body, serta agen silinder planar.
Mekanisme menarik/gripper belum ditambahkan. Arsitektur actor/critic mengikuti
baseline; dimensi aksi, observasi, dan state per agen tetap sama.

## Instalasi terisolasi di remote

Salin seluruh folder ini ke PC remote. Gunakan Python 3.11 jika tersedia
(Python 3.10 juga didukung dependensi yang dikunci), lalu dari root proyek baru:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Jika hanya ada Python 3.10, ganti `python3.11` dengan `python3.10` pada pembuatan
venv. Jangan memakai `.venv` proyek lama. PyTorch harus mendeteksi CUDA; status
driver pada `nvidia-smi` saja belum memverifikasi seluruh stack Python.
FFmpeg pada PATH diperlukan hanya untuk menyimpan video.

MuJoCo 3.5.0, mujoco-warp 3.5.0 dan Warp 1.11.0 dikunci sebagai pasangan API
yang eksplisit. Jangan memperbarui satu paket sendiri. Cache kompilasi Warp
disimpan di `.cache/warp/` proyek ini. JIT awal dapat membutuhkan waktu.

## Urutan pengujian remote

```bash
# 1. Dependensi dan ketersediaan CUDA; kode keluar nonzero jika belum siap.
.venv/bin/python -m tshape_mjwarp.cli check --json runs/check.json

# 2. Tes biasa, lalu tes CUDA eksplisit. Permintaan tes CUDA tanpa GPU akan gagal.
MUJOCO_GL=egl .venv/bin/python -m unittest discover -s tests -v
TSHAPE_RUN_GPU_TESTS=1 MUJOCO_GL=egl .venv/bin/python -m unittest discover -s tests -p test_gpu.py -v

# 3. Perbandingan lintasan singkat dan fitur terhadap C MuJoCo dengan state sama.
.venv/bin/python -m tshape_mjwarp.cli validate --num-envs 2 --json runs/validate.json

# 4. Throughput rollout setelah warm-up, tanpa PPO.
.venv/bin/python -m tshape_mjwarp.cli benchmark --counts 16 32 64 128 --warmup 10 --steps 100 --json runs/benchmark_rollout.json

# 5. Throughput pengumpulan data + update PPO.
.venv/bin/python -m tshape_mjwarp.cli benchmark --counts 16 32 64 128 --warmup 10 --steps 128 --with-updates --json runs/benchmark_training.json
```

Jalankan benchmark tanpa beban GPU lain untuk perbandingan yang bermakna.
Kirim `check.json`, `validate.json`, kedua benchmark, dan error bila ada.
`environment_transitions_per_second` menghitung keputusan tim dari seluruh arena;
`nominal_physics_substeps_per_second` adalah nilai nominal setelah dikalikan frame
skip, bukan jumlah substep sehat yang diukur satu per satu. Waktu kompilasi awal
dikeluarkan melalui warm-up. Rendering, video, dan evaluasi tidak masuk benchmark.
Benchmark `--with-updates` mencakup PPO tetapi tidak logging/checkpoint ke disk;
kecepatan wall-clock training lengkap dapat lebih rendah.

`torch_peak_allocated_mib` hanya mencakup alokasi PyTorch, bukan seluruh alokasi
Warp. `device_free_mib` adalah memori bebas GPU keseluruhan dan dipengaruhi proses
lain. Gunakan juga `nvidia-smi` saat benchmark untuk memantau pemakaian total.
Kegagalan/OOM dicatat sebagai `status: error`, bukan diubah menjadi angka throughput.

Untuk debugging CPU tanpa CUDA:

```bash
.venv/bin/python -m tshape_mjwarp.cli validate --backend reference --device cpu --num-envs 2
TSHAPE_RUN_WARP_CPU_TESTS=1 .venv/bin/python -m unittest discover -s tests -p test_warp_cpu.py -v
```

Tes GPU yang dilewati bukan hasil lulus GPU. Validasi lintasan singkat memakai
toleransi, bukan kesamaan bitwise, dan bukan bukti kemampuan policy belajar.

## Training

```bash
.venv/bin/python -m tshape_mjwarp.cli train --num-envs 32 --num-agents 8 --total-steps 1000000 --run-dir runs/medium_n8_e32_s42
```

`configs/environment.yaml` menggunakan arena sempit dengan bukaan 0,50 m dan
ketebalan sekat 0,03 m. Ubah XML di `assets/arena.xml`
untuk eksperimen baru pada proyek ini. Timestep 0,001 s, frame skip 50, gaya
per agen, massa, serta friction/damping mengikuti XML tersebut.

Opsi spawn independen:

```bash
.venv/bin/python -m tshape_mjwarp.cli train --num-envs 32 --num-agents 8 --randomize-agents --randomize-object-position --randomize-object-orientation --total-steps 1000000 --run-dir runs/medium_random_n8_e32_s42
```

Masing-masing memiliki bentuk `--no-randomize-...`. Rentang posisi dan yaw ada
di YAML environment. Jika agen tidak diacak, formasinya mengikuti pose objek.
RNG setiap arena terpisah dan tersimpan. Generator backend baru tidak identik
dengan NumPy baseline: seed sama tidak berarti layout sama antar-backend.
Untuk perbandingan fisika, validasi menyalin state awal yang sama secara eksplisit.
Sampling GPU memakai kandidat terbatas (128 pose, 512 kandidat per agen) dan
mengeluarkan error jika habis, bukan diam-diam menggunakan layout yang overlap.

`configs/training.yaml` memisahkan `simulation` dan `learning`. `num_envs` adalah
jumlah arena, bukan jumlah agen. Default 32 arena x 32 keputusan menghasilkan
1.024 transisi tim per update. Mengubah panjang rollout dapat mengubah temporal
GAE dan dinamika PPO meskipun ukuran batch total sama. `total_steps` harus habis
dibagi jumlah arena agar budget tidak dilewati diam-diam.

Pemeriksaan state finite, penetrasi, batas arena dan speed dilakukan setiap
substep. Arena gagal dibekukan/dipulihkan sambil arena sehat melanjutkan langkah.
Kapasitas kontak/constraint diperiksa; overflow harus diperbaiki dengan alokasi
yang sesuai, bukan mengabaikan kontak. MJWarp tidak menyediakan seluruh counter
warning C MuJoCo dengan antarmuka identik; validasi tambahan di remote tetap perlu.
Implementasi awal belum memakai CUDA graph gabungan physics/task/reset; benchmark
akan menentukan optimasi selanjutnya tanpa mengurangi pemeriksaan keselamatan.

## Evaluasi, video dan resume

```bash
.venv/bin/python -m tshape_mjwarp.cli evaluate runs/medium_n8_e32_s42/checkpoints/best.pt --episodes 100 --seed 2000000 --json runs/medium_n8_e32_s42/heldout_100.json --video-dir runs/medium_n8_e32_s42/videos/heldout

# Viewer pada mesin yang memiliki desktop/display:
MUJOCO_GL=glfw .venv/bin/python -m tshape_mjwarp.cli evaluate runs/medium_n8_e32_s42/checkpoints/latest.pt --episodes 1 --render

.venv/bin/python -m tshape_mjwarp.cli train --resume runs/medium_n8_e32_s42/checkpoints/latest.pt --total-steps 2000000 --run-dir runs/medium_n8_e32_s42_resume
```

Video merekam episode pertama saja sebagai `policy.mp4`. Render memakai state
aktual hasil simulasi, diteruskan ke renderer CPU MuJoCo; ini bukan rollout fisika
CPU pengganti. Evaluasi otomatis memakai seed monitoring 1.000.000 dan seterusnya;
evaluasi akhir sebaiknya memakai seed lain seperti contoh di atas.

Checkpoint baru menyimpan XML/hash, konfigurasi, model/optimizer, normalisasi,
RNG minibatch, Torch/CUDA, dan RNG spawn per arena. Formatnya berbeda dari baseline.
Checkpoint baseline tidak diterima secara otomatis. Resume memulihkan training,
tetapi memulai episode baru, bukan melanjutkan state fisika/rollout yang terpotong.
Backend, device, jumlah arena, randomisasi, seed, dan hyperparameter dipertahankan
saat resume; CLI menerima budget kumulatif baru. Run directory harus baru.

Output: `manifest.json`, snapshot XML/YAML, `episodes.jsonl`, `updates.jsonl`,
`evaluations.jsonl`, `evaluation_best.json`, `evaluation_latest.json`, serta
`checkpoints/best.pt` dan `latest.pt`. Best diurutkan menurut success rate,
kemudian waktu selesai yang lebih singkat, lalu mean return. Logging awal JSONL;
TensorBoard dan video otomatis selama training belum diaktifkan. Video tersedia
melalui evaluasi manual agar benchmark dan training tidak terbebani rendering.

## Struktur kode

- `config.py`, `training_config.py`: konfigurasi dan validasi.
- `physics.py`, `kernels.py`: backend MJWarp/reference serta reduksi kontak.
- `spawning.py`, `task.py`, `environment.py`: RNG, geometri, reward, fitur, reset.
- `rollout.py`, `normalization.py`, `mappo.py`: GAE dan PPO pada tensor device.
- `trainer.py`, `checkpoint.py`, `artifacts.py`: training dan hasil mandiri.
- `evaluation.py`, `video.py`, `validation.py`, `benchmark.py`, `cli.py`: tooling.
- `reference/`: oracle CPU dan actor/critic yang disalin; tanpa dependensi folder lama.

Referensi API: https://mujoco.readthedocs.io/en/stable/mjwarp/index.html
