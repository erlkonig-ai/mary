//! f5 — F5-TTS (flow-matching expressive TTS) ported to
//! Burn and held as a neural-network graph in TribleSpace. Input projection 712→1024 = mel⊕cond⊕
//! text(512); AdaLN-zero DiT (depth 22, dim 1024); ConvNeXt-V2 text encoder;
//! CFM Euler sampler; Vocos vocoder. Built on `mary::nn` + `mary::format`.

pub mod cfm;
pub mod config;
pub mod dit;
pub mod mel;
pub mod model;
pub mod text;
pub mod tokenizer;
pub mod vocos;
pub mod wav;
