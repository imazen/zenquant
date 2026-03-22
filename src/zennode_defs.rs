//! zenode node definitions for palette quantization.
//!
//! Defines [`Quantize`] with parameters for palette size, quality, and dithering.

extern crate alloc;
use alloc::string::String;
#[cfg(test)]
use alloc::vec::Vec;

use zennode::*;

/// Palette quantization with perceptual masking.
///
/// Reduces truecolor images to indexed-color palettes using AQ-informed
/// clustering in OKLab color space. The `quality` preset controls the
/// speed/quality tradeoff, and `dither_strength` controls error-diffusion
/// dithering intensity.
///
/// JSON API: `{ "max_colors": 256, "quality": "best", "dither_strength": 0.5 }`
/// RIAPI: `?quant.max_colors=256&quant.quality=best&quant.dither_strength=0.5`
#[derive(Node, Clone, Debug)]
#[node(id = "zenquant.quantize", group = Quantize, role = Quantize)]
#[node(tags("quantize", "palette", "indexed"))]
pub struct Quantize {
    /// Maximum palette size (2-256 colors).
    ///
    /// Controls the number of distinct colors in the output palette.
    /// Fewer colors means smaller file sizes but lower color fidelity.
    /// Most indexed formats (GIF, PNG, WebP lossless) support up to 256.
    #[param(range(2..=256), default = 256, step = 1)]
    #[param(unit = "colors", section = "Main", label = "Max Colors")]
    #[kv("quant.max_colors", "max_colors")]
    pub max_colors: i32,

    /// Quality preset: "fast", "balanced", or "best".
    ///
    /// - `"fast"` — no masking, histogram-only k-means (~25ms per 512x512)
    /// - `"balanced"` — AQ masking + 2 k-means iterations + greedy run extension
    /// - `"best"` — AQ masking + 8 k-means iterations + Viterbi DP (default)
    #[param(default = "best")]
    #[param(section = "Main", label = "Quality")]
    #[kv("quant.quality", "quality")]
    pub quality: String,

    /// Dithering strength (0.0 = none, 1.0 = full).
    ///
    /// Controls how aggressively error-diffusion dithering is applied.
    /// Higher values reduce color banding but add noise. Lower values
    /// produce cleaner output that compresses better.
    #[param(range(0.0..=1.0), default = 0.5, identity = 0.5, step = 0.05)]
    #[param(unit = "", section = "Main", label = "Dithering")]
    #[kv("quant.dither_strength", "dither_strength")]
    pub dither_strength: f32,
}

impl Default for Quantize {
    fn default() -> Self {
        Self {
            max_colors: 256,
            quality: String::from("best"),
            dither_strength: 0.5,
        }
    }
}

/// Registration function for aggregating crates.
pub fn register(registry: &mut NodeRegistry) {
    registry.register(&QUANTIZE_NODE);
}

/// All zenquant zenode definitions.
pub static ALL: &[&dyn NodeDef] = &[&QUANTIZE_NODE];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_metadata() {
        let schema = QUANTIZE_NODE.schema();
        assert_eq!(schema.id, "zenquant.quantize");
        assert_eq!(schema.group, NodeGroup::Quantize);
        assert_eq!(schema.role, NodeRole::Quantize);
        assert!(schema.tags.contains(&"quantize"));
        assert!(schema.tags.contains(&"palette"));
        assert!(schema.tags.contains(&"indexed"));
    }

    #[test]
    fn param_count_and_names() {
        let schema = QUANTIZE_NODE.schema();
        let names: Vec<&str> = schema.params.iter().map(|p| p.name).collect();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"max_colors"));
        assert!(names.contains(&"quality"));
        assert!(names.contains(&"dither_strength"));
    }

    #[test]
    fn defaults() {
        let node = QUANTIZE_NODE.create_default().unwrap();
        assert_eq!(node.get_param("max_colors"), Some(ParamValue::I32(256)));
        assert_eq!(
            node.get_param("quality"),
            Some(ParamValue::Str(String::from("best")))
        );
        assert_eq!(
            node.get_param("dither_strength"),
            Some(ParamValue::F32(0.5))
        );
    }

    #[test]
    fn json_round_trip() {
        let mut params = ParamMap::new();
        params.insert("max_colors".into(), ParamValue::I32(128));
        params.insert("quality".into(), ParamValue::Str("fast".into()));
        params.insert("dither_strength".into(), ParamValue::F32(0.3));

        let node = QUANTIZE_NODE.create(&params).unwrap();
        assert_eq!(node.get_param("max_colors"), Some(ParamValue::I32(128)));
        assert_eq!(
            node.get_param("quality"),
            Some(ParamValue::Str("fast".into()))
        );
        assert_eq!(
            node.get_param("dither_strength"),
            Some(ParamValue::F32(0.3))
        );

        // Round-trip through to_params/create
        let exported = node.to_params();
        let node2 = QUANTIZE_NODE.create(&exported).unwrap();
        assert_eq!(node2.get_param("max_colors"), Some(ParamValue::I32(128)));
        assert_eq!(
            node2.get_param("quality"),
            Some(ParamValue::Str("fast".into()))
        );
        assert_eq!(
            node2.get_param("dither_strength"),
            Some(ParamValue::F32(0.3))
        );
    }

    #[test]
    fn from_kv_prefixed() {
        let mut kv = KvPairs::from_querystring(
            "quant.max_colors=64&quant.quality=balanced&quant.dither_strength=0.8",
        );
        let node = QUANTIZE_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("max_colors"), Some(ParamValue::I32(64)));
        assert_eq!(
            node.get_param("quality"),
            Some(ParamValue::Str("balanced".into()))
        );
        assert_eq!(
            node.get_param("dither_strength"),
            Some(ParamValue::F32(0.8))
        );
        assert_eq!(kv.unconsumed().count(), 0);
    }

    #[test]
    fn from_kv_alias() {
        let mut kv = KvPairs::from_querystring("max_colors=32&quality=fast&dither_strength=0.1");
        let node = QUANTIZE_NODE.from_kv(&mut kv).unwrap().unwrap();
        assert_eq!(node.get_param("max_colors"), Some(ParamValue::I32(32)));
        assert_eq!(
            node.get_param("quality"),
            Some(ParamValue::Str("fast".into()))
        );
        assert_eq!(
            node.get_param("dither_strength"),
            Some(ParamValue::F32(0.1))
        );
    }

    #[test]
    fn from_kv_no_match() {
        let mut kv = KvPairs::from_querystring("w=800&h=600");
        let result = QUANTIZE_NODE.from_kv(&mut kv).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn downcast_to_concrete() {
        let node = QUANTIZE_NODE.create_default().unwrap();
        let q = node.as_any().downcast_ref::<Quantize>().unwrap();
        assert_eq!(q.max_colors, 256);
        assert_eq!(q.quality, "best");
        assert!((q.dither_strength - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn registry_integration() {
        let mut registry = NodeRegistry::new();
        register(&mut registry);
        assert!(registry.get("zenquant.quantize").is_some());

        let result = registry.from_querystring("quant.max_colors=128&quant.quality=balanced");
        assert_eq!(result.instances.len(), 1);
        assert_eq!(result.instances[0].schema().id, "zenquant.quantize");
    }
}
