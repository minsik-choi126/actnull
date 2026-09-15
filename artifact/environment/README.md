# Environment record

`pyproject.toml` and `uv.lock` are the producer package specification and exact
resolver lock retained with the v7 rerun code. The lock includes the optional
data and figure dependency groups and platform-specific alternatives.

`runtime-freeze.txt` is the exact set of third-party distributions installed in
the Python 3.11.16 CPU environment used for the shipped v7 numerical reruns. It
was captured with `uv pip freeze`. The local editable producer package does not
appear in that list, and optional dataset/figure packages that were not loaded by
the numerical reruns were not installed in this particular environment.

The alternative calibration-composition materializer was a separate data-stage
run. Its result map recorded `datasets==5.0.1` and `pillow==12.3.0`, but a full
installed transitive freeze was not retained. This is stated explicitly in
`composition-materialization-runtime.json`; that file does not pretend the
numerical freeze was the data-stage environment. The resolver lock is the exact
release reconstruction record for the optional `data` group and includes the
transitive packages needed to materialize the control.

The historical environment for superseded archived v6 commands was not
recoverable as an exact freeze. These records pin the numerical environment that
generated and checked the v7 row-level results, the versions directly observed
during composition materialization, and a locked reconstruction environment.
They do not claim to recover unrecorded transitive packages from the older or
data-stage environments.

To validate that the specification and lock agree (with `uv` installed), run:

```bash
uv lock --check --project artifact/environment
```

The result checker itself uses only the Python standard library.
