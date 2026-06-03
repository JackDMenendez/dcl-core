# Reproducing This Paper's Experiments

This repository includes a Docker image that freezes the Python environment used to generate all results in the paper. This ensures that the experiments can be reproduced with exact package versions, without relying on the authors' development environment or manual package installation.

## For Reviewers

### Prerequisites
- **Docker** or **Podman** installed
- A copy of this repository
- (Optional) VS Code with extensions from `.vscode/extensions.txt`

### Setting Up VS Code (Optional)

If `.vscode/extensions.txt` is present, you can reproduce the exact VS Code environment used during development:

```bash
cat .vscode/extensions.txt | xargs -n 1 code --install-extension
```

This is optional—the experiments will run in the container regardless of your VS Code setup. Extensions are only needed if you want to browse or edit the source code with the same tools the authors used.

### Building the Image

```bash
docker build -t paper-experiments:latest .
```

This reads `requirements.txt` and installs all pinned package versions into the image.

### Running Experiments

Replace `experiments/run.py` with the actual script path and arguments:

```bash
# Using the provided runner script
bash shells/bash/tools/run-container.sh python experiments/run.py --case baseline

# Or directly with docker
docker run --rm -v $(pwd):/workspace paper-experiments:latest python experiments/run.py --case baseline
```

The image mounts the repository at `/workspace` inside the container, so all paths resolve correctly.

### Verifying Results

Results should match the paper's reported values. If you encounter differences:

1. Check that you built the image with the correct Dockerfile (default is `.dev-shell/Dockerfile`)
2. Verify the Python version inside the image: `docker run paper-experiments:latest python --version`
3. Confirm all packages installed: `docker run paper-experiments:latest pip list`

If issues persist, contact the authors with your Docker version and OS.

## For Authors

### Creating the Reproducible Image

The `.dev-shell/` directory and `.vscode/` directory contain configuration for the reproducible environment:

- **Dockerfile**: Template for building the reproducible image
- **requirements.txt**: Pinned package versions from your development environment
- **container.conf**: Configuration for the dev-shell runners (optional for reviewers)

The `.vscode/extensions.txt` file (if present) documents the VS Code extensions used during development:

- **extensions.txt**: Frozen list of VS Code extensions with versions (optional for reviewers)

### Workflow

1. **During development**: Work normally in your Python venv with whatever versions you're using.

2. **Before paper release**:
   - Ensure your experiments run correctly in your current venv
   - Run the `generate-dockerfile` helper to snapshot your environment:
     ```bash
     bash shells/bash/lib/generate-dockerfile.sh .
     ```
     or on Windows:
     ```cmd
     call shells\windows\lib\generate-dockerfile.cmd .
     ```
   - This creates `.dev-shell/requirements.txt` with all pinned versions
   - (Optional) Export VS Code extensions if reviewers might want to recreate your editing environment:
     ```bash
     bash shells/bash/tools/export-vscode-extensions.sh .
     ```
     or on Windows:
     ```cmd
     call shells\windows\tools\export-vscode-extensions.cmd .
     ```
   - This creates `.vscode/extensions.txt` (not required for running experiments)

3. **Customize the Dockerfile** (if needed):
   - The generated Dockerfile uses `python:3.12-slim` as a base
   - Edit `.dev-shell/Dockerfile` to change the Python version or add system dependencies
   - Test it locally: `docker build -t paper-experiments:latest .`

4. **Release**:
   - Commit `.dev-shell/Dockerfile` and `.dev-shell/requirements.txt` to git
   - Tag the release in your repository (e.g., `paper-v1.0`)
   - Include instructions for reviewers (this file serves that purpose)

### Testing the Image

Before submitting the paper, verify that experiments run inside the image:

```bash
docker build -t paper-experiments:latest .
docker run --rm -v $(pwd):/workspace paper-experiments:latest python experiments/run.py --case baseline
```

Compare the results to those in your paper. If they differ, troubleshoot before submission.

### Why Docker?

- **Reproducibility**: Exact Python version and package versions
- **Portability**: Reviewers don't need your development environment
- **Immutability**: Image hash is deterministic; same image, same results
- **No manual setup**: Single `docker build` command instead of manual installation

### What About the Dev Environment?

The `shells/bash/` and `shells/windows/` directories are for **local development only**. Reviewers don't need them. They only need:
- The repository source code
- The Dockerfile and requirements.txt
- Docker or Podman installed

## Troubleshooting

**Docker build fails:**
- Ensure `requirements.txt` exists in `.dev-shell/`
- Check that all packages in the file are available on PyPI
- Some packages may require compiled dependencies; add `RUN apt-get install ...` to the Dockerfile

**Experiments run but results differ from paper:**
- Verify Python version matches (check paper Methods section)
- Check for non-deterministic elements (random seeds, floating-point precision)
- Ensure all input data files are in the repo

**Image is too large:**
- Use a smaller base image (e.g., `python:3.12-slim` instead of `python:3.12`)
- Remove development packages: add `pip install --no-cache-dir` to the Dockerfile

**Can't run inside container:**
- Ensure the command path is correct (use `/workspace` as the working directory inside the container)
- Check file permissions if running on Linux/Mac
- Try running a simple command first: `docker run paper-experiments:latest python --version`
