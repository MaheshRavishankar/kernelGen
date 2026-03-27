# kernelGen

A framework for benchmarking and comparing AMD GPU kernel providers
across operations (GEMM, attention, convolutions, etc.).

See [docs/InitialDesign.md](docs/InitialDesign.md) for motivation,
project layout, and usage instructions.

## Quick start

```bash
# Python setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Configure and build
cmake --preset Release
cmake --build ~/kernelGen/build/Release

# Generate test data and run
.venv/bin/python gemm/tests/generate_test.py gemm/tests/small_f16/config.json
.venv/bin/python gemm/kernel_providers/hipblaslt/run.py --test-dir gemm/tests
```

## License

[MIT](LICENSE)
