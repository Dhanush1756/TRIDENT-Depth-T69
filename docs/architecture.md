# Implemented application and model architecture

```mermaid
flowchart TD
    UI[HTML CSS JavaScript: capabilities, uploads, comparisons] --> API[FastAPI: units, size, mask and alignment validation]
    API --> Dispatch[ApplicationEngine: method and protocol guard]
    Dispatch --> CV[Nearest / bilinear / bicubic / guided / joint bilateral / classical]
    Dispatch --> Legacy[Unchanged legacy inference: B0 deep dense4 only]
    Dispatch --> Exact[SHA256-verified extended epoch12 checkpoint]
    Exact --> Init[Bicubic interpolation + nearest valid hole fill]
    RGB[Registered target RGB] --> Features[RGB / metric initializer / support / scale / protocol / RGB-present]
    Init --> Features
    Features --> CNN[32-64-128 encoder, skips, nearest decoder, residual depth]
    CNN --> Deep[Raw extended depth]
    CNN --> Boundary[Supervised depth-boundary head]
    Deep --> Hybrid[Original extended classical/geometry fusion]
    Boundary --> Hybrid
    CV --> Hybrid
    Hybrid --> Consistency[Dense block means / clean observed points / no noisy hard copy]
    CV --> Output[Metric depth, masks and intermediates]
    Legacy --> Output
    Deep --> Output
    Consistency --> Output
    Output --> API
    API --> Preview[Shared-range previews / NPZ metres / optional PNG millimetres]
    Preview --> UI
    GT[Optional ground truth] --> Metrics[Original reference-masked metrics]
    Output --> Metrics
```

Dense SR: 2x/4x/8x. Completion: unchanged target grid, canonical learned conditioning scale 4.
The unused extended uncertainty head is frozen, not used as confidence. Geometry support and
boundary supervision mitigate texture artifacts but do not prove their absence. The app does not
run training or access any dataset split. Learned functionality requires an actual configured
checkpoint. Exact epoch-12 app inference has passed local synthetic HTTP/browser checks;
public Modal HTTP/browser checks passed. Final independent scientific benchmark results remain pending.

Training/data/benchmark code is preserved byte-for-byte during this application phase so the
externally running Colab benchmark and checkpoint source identity remain intact.
