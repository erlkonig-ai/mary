# Changelog

## Unreleased

- Pin the build root to AnyBytes `066c32a7`, matching TribleSpace, Faculties,
  and Drive. Temporary section freezes no longer perform durability flushes;
  explicit `ByteArea::persist` owns that barrier. Model identities, tensor
  encodings, numerical kernels, and feature selections are unchanged.
