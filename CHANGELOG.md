# Changelog

All notable changes to the "Colmena Cuántica" ecosystem will be documented here.

## [1.0.0] - 2026-01-20 (Gold Master)
### Added
- **Protocol Fourier**: Implementation of Bio-Spectral Resonance architecture.
- **Fractal Tensors**: Multi-resolution input space ($\mathbb{R}^{27}$) combining Micro (1m), Meso ($T_i$), and Macro (4h) layers.
- **Genetic Time Dilation**: Agents now possess a unique gene $T_i \in [10, 60]$ defining their perception of time.
- **Atomic Persistence**: Checkpoint saving is now atomic via `os.replace` to prevent corruption.
- **Health Monitor**: New `/health` endpoint in API.
- **Scientific Documentation**: Added `MATH_SPEC`, `BIO_SPEC`, `TECH_SPEC`.

### Changed
- **Ticker Universe**: Reduced to Top 10 High-Liquidity assets for focused learning.
- **Network Topology**: Policy Network upgraded to `27 -> 11` dimensions.
- **Training Engine**: `GymEngine` now constructs tensors on-the-fly (lazy evaluation).

### Removed
- Legacy support for static frequency analysis.
- `src/isaac_sim` and `src/nlp_service` (dead code).
- Deprecated 100d vector checkpoints.

---
## [0.9.0] - 2026-01-18 (Beta)
### Added
- Sentiment Analysis via VAE Latent Space ($z \in \mathbb{R}^8$).
- Initial Docker Infrastructure.

### Fixed
- Fixed recursion error in `src` directory structure.
- Stabilized `tqdm` progress bars in Docker output.
