# Installing AlphaFold with the Triton patch

This guide installs AlphaFold from Git and applies the standalone Triton/ROCm
patch. It assumes a Linux host with an AMD GPU, ROCm 7.1-compatible drivers,
Docker, and permission to access `/dev/kfd` and `/dev/dri`.

## 1. Clone AlphaFold

Choose a directory for the checkout and clone the upstream repository:

```bash
git clone https://github.com/deepmind/alphafold.git
cd alphafold
```

The patch was created against the AlphaFold baseline used by this project. To
apply it to the current checkout, run:

```bash
git apply /path/to/patches/alphafold-triton-rocm.patch
```

If the patch is downloaded from this repository instead, use its raw file URL:

```bash
curl -L \
  https://raw.githubusercontent.com/SarahBeecroft/alphafold/main/patches/alphafold-triton-rocm.patch \
  -o /tmp/alphafold-triton-rocm.patch
git apply /tmp/alphafold-triton-rocm.patch
```

If `git apply` reports that the upstream checkout is too different, clone the
matching project revision and apply the patch there:

```bash
git checkout c77e5d2
git apply /path/to/patches/alphafold-triton-rocm.patch
```

## 2. Download AlphaFold databases and parameters

Install the host-side download tools:

```bash
sudo apt-get update
sudo apt-get install -y aria2 rsync
```

Download the databases into a directory outside the Git checkout. This can
require hundreds of gigabytes of storage:

```bash
export DOWNLOAD_DIR=/data/alphafold
scripts/download_all_data.sh "$DOWNLOAD_DIR"
```

Do not place `DOWNLOAD_DIR` inside the repository, otherwise Docker may include
the databases in its build context.

## 3. Build the patched container

Build from the repository root. The final `.` is required because it is the
Docker build context containing the patched source:

```bash
docker build \
  -f docker/Dockerfile.triton \
  -t alphafold-triton .
```

The Dockerfile installs the ROCm JAX wheels, Triton, `jax-triton`, AlphaFold's
Python dependencies, HH-suite, HMMER, and Kalign. It also verifies that the
patched Triton source and attention dispatch are present.

## 4. Run a prediction

Place a FASTA file in the checkout, or use an absolute path to one elsewhere:

```bash
export FASTA_PATH="$PWD/your_protein.fasta"
export OUTPUT_DIR="$PWD/alphafold_output"
mkdir -p "$OUTPUT_DIR"
```

Run the image with the AMD GPU devices exposed:

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e AF2_USE_TRITON=1 \
  -v "$DOWNLOAD_DIR:/data:ro" \
  -v "$PWD:/work" \
  alphafold-triton \
  --fasta_paths="/work/$(basename "$FASTA_PATH")" \
  --data_dir=/data \
  --output_dir=/work/alphafold_output \
  --max_template_date=2022-01-01
```

The predicted structures and logs will be written to `alphafold_output`.

## 5. Disable Triton if needed

The patched code retains the original JAX attention implementation. To disable
the Triton path for a run:

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e AF2_USE_TRITON=0 \
  -v "$DOWNLOAD_DIR:/data:ro" \
  -v "$PWD:/work" \
  alphafold-triton \
  --fasta_paths=/work/your_protein.fasta \
  --data_dir=/data \
  --output_dir=/work/alphafold_output \
  --max_template_date=2022-01-01
```

## 6. Troubleshooting

Check that the host can see the AMD GPU before running AlphaFold:

```bash
rocminfo
```

If Docker cannot access the devices, check that the user is in the `video`
group and that the ROCm container prerequisites are installed. Triton is
enabled only when both `AF2_USE_TRITON=1` and the Triton Python packages are
available.

The existing [`docker/Dockerfile`](../docker/Dockerfile) is a separate
NVIDIA/CUDA image and should not be used for this ROCm/Triton setup.
