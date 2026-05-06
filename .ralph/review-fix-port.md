# Review and Fix ZAYA MLX Port

Review the current hand-written MLX ZAYA1-8B port for correctness and safety without running full model generation, then fix concrete issues with small, frequent commits.

## Goals
- Audit implementation against local Zyphra Transformers source/model config where possible.
- Avoid running full MLX generation/load paths that crash or time out.
- Fix clear correctness, safety, and UX issues in the port.
- Add lightweight verification that does not load the 8B model weights.
- Commit frequently with descriptive messages.

## Checklist
- [x] Inspect current git state and code structure.
- [x] Locate installed Zyphra Transformers ZAYA implementation for comparison.
- [x] Compare config/model assumptions against upstream implementation.
- [x] Identify no-full-load verification strategy.
- [ ] Fix import/path robustness for chat script if needed.
- [x] Make chat startup validation optional/safe.
- [x] Add lightweight architecture/key/shape inspection tooling.
- [x] Add tests or compile checks that avoid full model execution.
- [ ] Update README with safety and validation notes.
- [ ] Commit each coherent change.

## Verification
- `uv run python -m py_compile scripts/*.py` passed.
- `uv run python scripts/inspect_zaya_port.py --local-files-only` passed: 2483 HF indexed tensors, 2483 MLX parameters, 0 missing, 0 extra. Does not load full tensors.
- Enhanced inspection now also validates shapes after conv sanitize rules: 0 mismatches.
- Commits: `5445b04 Add safe ZAYA port inspection`, `d758f2b Validate MLX port tensor shapes safely`, `fbb0ff6 Reduce MLX expert routing memory spike`.

## Notes
- User explicitly said not to run the model because it crashes/times out.
- Located upstream implementation at `.venv/lib/python3.13/site-packages/transformers/models/zaya/modeling_zaya.py`.
