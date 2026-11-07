# Public history cleanup verification

This records the local verification completed before the owner authorized publishing the cleaned history and code. Earlier preparation plans/projection records are retained as dated evidence.

Both main and jathin were based on the reviewed old tip `633b55b95c636f6b39ec413cd847b3c3dbd7c8a7`. Its sanitized replacement is `58bf70da990e0fbc74cc3f29cc268645b022e651`. The signed initial commit `1ee87ec21fc023d21adb46b4d617d4a331fc46be` is byte-identical. A separate release commit adds the reviewed application and project files.

Actual cleaned-history verification: two historical commits, 199 reachable objects, 174 decoded text objects; zero detected home paths, absolute drive paths or known credential signatures; four withheld image blobs absent; Git full/strict integrity passed. All 83 protected historical scientific files are unchanged. The current 104 protected source/config/notebook files, nine evidence pairs and private checkpoint SHA256 passed preservation checks. No training or official-test evaluation occurred.

All 295 tests passed in both original and cleaned release copies (289 existing plus six privacy tests), with one existing Starlette/httpx deprecation warning. Four existing context-whitespace warnings in the preserved diagnostic patch were retained rather than modifying evidence.

Original private evidence and the verified complete repository backup are unchanged and must never be published. Pattern scans do not guarantee absence of all unknown secret formats. Rewriting branches does not erase external clones or platform caches.

Collaborators must fresh-clone after the coordinated public update. Never merge or push old history back; transfer only reviewed diffs and rescan them. Both affected branches require cleanup, rather than updating main alone.

The deployment workflow is manual-only for this release and remains owner-gated. No Modal operation or deployment is part of publication; automatic deployment integration requires a later instruction. Scientific evaluation status remains 854/861 saved task/frame records, with sparse_noisy_holes 116/123.
