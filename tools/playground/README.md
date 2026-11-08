# Mora-3 local model playground

From the repository root, run `python3 tools/playground/serve.py` and open the printed `http://127.0.0.1:8632/playground.html` URL. This preview uses the Python standard library and offers synthetic examples and result inspection. The launcher serves the bundled page directly and uses the stylesheet in `docs/assets/`.

To run a compatible checkpoint, use an existing Python environment with Mora-3’s PyTorch and Pillow dependencies. Keep the checkpoint folder and its `manifest.json` and `checkpoint.pt` files outside Git. Obtain the manifest SHA-256 from the checkpoint publisher through a trusted release channel and verify your download. A qualified full-model checkpoint release is pending. The [project repository](https://github.com/amelie-iska/Mora-3) is the source for release updates.

```bash
PYTHONPATH=src:tools/playground MORA3_PLAYGROUND_MANIFEST_SHA256='<published manifest SHA-256>' python tools/playground/serve.py --adapter native_playground:factory --weights-dir '/path/to/compatible-checkpoint'
```

Open the printed URL, select the checkpoint directory and choose **Load checkpoint**. The page shows the loaded model, checkpoint fingerprint and available outputs. `--weights-root '/path/to/checkpoints'` lists immediate child directories. The native adapter accepts only the manifest matching the pinned SHA-256 for that launch and also checks the source snapshot, configuration and weight identity.

The current native route is a foreground diagnostic. It accepts one 8-bit grayscale PNG up to 64 × 64 pixels and 64 KiB with an empty instruction and conversation. Choose **Predict or annotate** and **Foreground probability raster**, select **Microscopy or mask** for the attachment, and choose **Run model**. The result is a raster in the original frame with the checkpoint provenance. The remaining output formats are available as labeled synthetic examples.

The browser sends model requests and attached files to the loopback service only when you choose **Run model**. The service reads the weights directory locally. It does not accept cross-origin requests. See the [serving contract](playground-contract.md) for the adapter and result formats.
