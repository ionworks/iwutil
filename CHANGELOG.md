# Changelog — iwutil

All notable changes to this package are documented here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this package follows [Semantic Versioning](https://semver.org/).

For platform-wide release notes (Studio, pipeline, SDK, and more),
see [docs.ionworks.com/changelog](https://docs.ionworks.com/changelog).

<!-- New release sections are prepended below by the release-packages skill. -->

## [0.5.4] - 2026-08-03

### Changed
- Raised the `pandas` lower bound in the `parquet`/`feather` extras to
  `>=3.0.5`, and the `matplotlib` lower bound to `>=3.11.1`.

## [0.5.3] - 2026-07-29

### Changed
- Raised the `polars` lower bound in the `polars` extra to `>=1.43.0`.

## [0.5.2] - 2026-07-24

### Changed
- Migrated to numpy 2 and pandas 3.

### Fixed
- Incremented the numpy lower pin and corrected MCMC bin syntax.

## [0.5.1] - 2026-07-20

### Changed
- Raised the optional ``polars`` extra floor from ``>=1.33.1`` to
  ``>=1.42.1`` (#1284).

## [0.5.0] - 2026-07-13

### Breaking changes
- Dropped support for Python 3.10; the minimum supported version is now
  Python 3.11 (#1174).

## [0.4.3] - 2026-06-01

### Changed
- Relaxed dependency pins: removed the `numpy<2` upper bound (now
  `numpy>=1.26`) and raised the `matplotlib` floor to `>=3.9` (#764).
- Switched the optional `polars` extra from `polars-lts-cpu` to the
  standard `polars` distribution (#768).
