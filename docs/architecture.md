# Implemented architecture

```mermaid
flowchart TD
    NYU[Official labelled NYU HDF5 and split MAT] --> Convert[Inspect axes / stream exact metric exports / scene audit]
    Convert --> Manifest
    Manifest[Scene-disjoint JSON manifest] --> Audit[Split and input validation]
    Files[RGB / numeric depth / optional mask] --> Audit
    Config[JSON configuration / explicit units] --> Audit
    Audit --> Sample[Aligned metric DepthSample]
    Sample --> A[Dense SR: area mean at 2 / 4 / 8]
    Sample --> B[Sparse completion: sampling / holes / noise]
    A --> Observation[Input depth and observation mask]
    B --> Observation
    Observation --> Classical[Interpolation / median / fill / joint bilateral]
    Sample --> RGB[Target-resolution RGB guidance]
    RGB --> Classical
    Classical --> Geometry[RGB gradients validated by depth jumps]
    Geometry --> Edge[Edge-aware depth candidate]
    Classical --> CNN[Baseline CNN OR residual encoder-decoder]
    RGB --> CNN
    Observation --> CNN
    CNN --> Learned[Residual-corrected depth and optional uncertainty]
    Classical --> Fusion[Normalized trust weights]
    Edge --> Fusion
    Learned --> Fusion
    Fusion --> Consistency[Hard or soft observed-value consistency]
    Consistency --> Output[Metric-depth output]
    Output --> Evaluation[Masked metrics / benchmark / inference ablations]
    Output --> API[FastAPI previews and numeric downloads]
    UI[HTML / CSS / JavaScript] --> API
```

The API permits classical methods without PyTorch. Deep/hybrid calls require a
trained checkpoint and fail explicitly otherwise. The CNN and fusion paths have
been tested structurally; they are not yet a trained reconstruction system.
The diagram shows implemented interfaces, not measured improvement.

The custom network uses 8 input channels: RGB (3), initial classical depth (1),
observed validity (1), scale/8 (1), sparse-mode indicator (1), RGB-present flag (1).
Encoder channels are 32/64/128; decoder channels 64/32, with interpolation and
skip connections supporting odd target sizes. Outputs are residual, clamped
Laplace log scale and optional sigmoid boundary confidence. Fusion currently
uses deterministic geometric boundary confidence, not the optional learned
boundary head. A separate three-layer 32-channel CNN is the named lightweight
baseline. Neither model is described as DKN/FDKN.
