# Changelog — iwutil

All notable changes to this package are documented here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this package follows [Semantic Versioning](https://semver.org/).

For platform-wide release notes (Studio, pipeline, SDK, and more),
see [docs.ionworks.com/changelog](https://docs.ionworks.com/changelog).

<!-- New release sections are prepended below by the release-packages skill. -->

## [0.4.3] - 2026-06-01

### Changed
- Relaxed dependency pins: removed the `numpy<2` upper bound (now
  `numpy>=1.26`) and raised the `matplotlib` floor to `>=3.9` (#764).
- Switched the optional `polars` extra from `polars-lts-cpu` to the
  standard `polars` distribution (#768).
