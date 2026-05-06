# Review and Fix ZAYA MLX Port

Review the current hand-written MLX ZAYA1-8B port for correctness and safety without running full model generation, then fix concrete issues with small, frequent commits.

## Goals
- Audit implementation against local Zyphra Transformers source/model config where possible.
- Avoid running full MLX generation/load paths that crash or time out.
- Fix clear correctness, safety, and UX issues in the port.
- Add lightweight verification that does not load the 8B model weights.
- Commit frequently with descriptive messages.

## Checklist
- [ ] Inspect current git state and code structure.
- [ ] Locate installed Zyphra Transformers ZAYA implementation for comparison.
- [ ] Compare config/model assumptions against upstream implementation.
- [ ] Identify no-full-load verification strategy.
- [ ] Fix import/path robustness for chat script if needed.
- [ ] Make chat startup validation optional/safe.
- [ ] Add lightweight architecture/key/shape inspection tooling.
- [ ] Add tests or compile checks that avoid full model execution.
- [ ] Update README with safety and validation notes.
- [ ] Commit each coherent change.

## Verification

## Notes
- User explicitly said not to run the model because it crashes/times out.
