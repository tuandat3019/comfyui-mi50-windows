# Optional CPU CI

`checks.yml` is a GitHub Actions template. To activate it, copy it to `.github/workflows/checks.yml` using an account/token permitted to manage workflows. The template installs Python3.12 and CPU Torch2.7.1, then runs the unit tests; it does not start GPU inference.

The initial publishing OAuth credential has repository access but lacks the `workflow` scope. Keeping this template outside `.github/workflows` allows the source repository to be published without requesting additional account permissions. The local CPU tests passed on Torch2.14.0+cpu and Torch2.7.1+cu118 with all GPUs hidden.
