//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the VIctory build
//! that translated it (the BLAKE3 of our executable), so a rebuilt
//! translator never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies this build: the BLAKE3 of our executable file and of
/// the translation settings it runs with (the loop limit, which a
/// diagnostics build can change at run time).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let exe = std::env::current_exe().and_then(std::fs::read);
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        match exe {
            Ok(mut bytes) => {
                bytes.extend_from_slice(&limit.to_le_bytes());
                // Include wave/f16 switches so toggling them invalidates cached translations.
                for name in [
                    "VICTORY_WAVE_WIDE",
                    "VICTORY_WAVE_DEEP_SKIPS",
                    "VICTORY_WAVE_OWN_BITS",
                    "VICTORY_F16_NEAREST",
                ] {
                    bytes.extend_from_slice(name.as_bytes());
                    bytes.push(b'=');
                    if let Some(value) = std::env::var_os(name) {
                        bytes.extend_from_slice(value.as_encoded_bytes());
                    }
                    bytes.push(0);
                }
                // And which clean-up passes ran over the translated code.
                bytes.extend_from_slice(&victory_shader::emit::shader_opt().to_le_bytes());
                // Renderer port 5 (`VICTORY_PORT_CBV=1`): constant buffers read as raw
                // bytes. Only when on, so translations cached without it keep their key.
                if victory_shader::emit::constants_raw() {
                    bytes.extend_from_slice(b"VICTORY_PORT_CBV=1");
                }
                if victory_shader::emit::raw_views() != 0 {
                    bytes.extend_from_slice(b"RAW_VIEWS=");
                    bytes.extend_from_slice(&victory_shader::emit::raw_views().to_le_bytes());
                }
                *blake3::hash(&bytes).as_bytes()
            }
            Err(e) => {
                // Without it no entry can be trusted: a fresh key per run.
                tracing::warn!(error = %e, "cannot read our executable; shader cache unused");
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                *blake3::hash(&now.as_nanos().to_le_bytes()).as_bytes()
            }
        }
    })
}

/// The translation of a console shader container, from the cache when
/// there is one, along with whether it uses XCL's root-signature ABI
/// (`u0 io`, `u1 res`, `u2..u5 mem`).
///
/// Uses `victory-shader` first (matching VIctory's root signature ABI), and
/// falls back to XCL's `gpu::recomp` / `gpu::dxil` backend when
/// `victory-shader` cannot translate a program (or when `VICTORY_XCL_SHADERS=1`).
pub(crate) fn host_shader_with_abi(container: &[u8]) -> Result<(HostShader, bool), String> {
    let translate = |c: &[u8]| translate_with_xcl_fallback(c);
    match CACHE.get() {
        Some(cache) => cached(cache, build(), container, translate),
        None => translate(container),
    }
}

fn translate_with_xcl_fallback(container: &[u8]) -> Result<(HostShader, bool), String> {
    let prefer_xcl = std::env::var("VICTORY_XCL_SHADERS").is_ok_and(|v| v == "1");
    if !prefer_xcl {
        match victory_shader::emit::translate(container) {
            Ok(shader) => return Ok((shader, false)),
            Err(err) => {
                if let Ok(shader) = translate_via_xcl(container) {
                    tracing::debug!("shader translated via XCL backend fallback (after victory-shader: {err})");
                    return Ok((shader, true));
                }
                return Err(err.to_string());
            }
        }
    }
    translate_via_xcl(container)
        .map(|s| (s, true))
        .or_else(|_| {
            victory_shader::emit::translate(container)
                .map(|s| (s, false))
                .map_err(|e| e.to_string())
        })
}

#[cfg(feature = "xcl-shaders")]
fn translate_via_xcl(container: &[u8]) -> Result<HostShader, String> {
    use xcl::gpu::recomp::pso::{self, Stage};
    let parsed = xcl::gpu::xbox::parse(container)
        .ok_or_else(|| "not an Xbox Series shader container".to_string())?;
    let hw = parsed
        .fssm
        .iter()
        .flat_map(|f| f.stages.iter())
        .map(|s| s.hw)
        .next()
        .ok_or_else(|| "no hardware stage in FSSM".to_string())?;
    let (stage, xcl_stage) = match hw {
        4 => (HwStage::Vs, Stage::Vertex),
        5 => (HwStage::Ps, Stage::Pixel),
        6 => (HwStage::Cs, Stage::Compute),
        7 => (HwStage::Ms, Stage::Compute),
        other => return Err(format!("unsupported hw stage {other} for XCL fallback")),
    };
    let split_wave64 = crate::host::host().is_some_and(|h| match xcl_stage {
        Stage::Compute => pso::wave32_only_for_device(&h.device),
        Stage::Vertex | Stage::Pixel => pso::graphics_wave32_for_device(&h.device),
    });
    let compiled = pso::console_to_dxil_for(container, xcl_stage, 0, split_wave64)?;
    let samplers: Vec<[u32; 4]> = compiled.samplers.iter().map(|&(w, _)| w).collect();
    let static_samplers = samplers.iter().copied().map(StaticSampler::from_words).collect();
    Ok(HostShader {
        stage,
        dxil: compiled.dxil,
        user_data: Vec::new(),
        samplers,
        static_samplers,
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    })
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<(HostShader, bool), String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        2
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let (shader, xcl_abi) = (self.translate)(&input[32..]).map_err(anyhow::Error::msg)?;
        Ok(encode(&shader, xcl_abi))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<(HostShader, bool), String> + Send + Sync,
) -> Result<(HostShader, bool), String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

const MAGIC: &[u8; 4] = b"VSH2";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

#[cfg(not(feature = "xcl-shaders"))]
fn translate_via_xcl(_container: &[u8]) -> Result<HostShader, String> {
    Err("xcl-shaders feature is disabled".to_string())
}

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader, xcl_abi: bool) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.root_indexed_views.len() as u32);
    for view in &s.root_indexed_views {
        for value in [
            u32::from(view.table_slot),
            u32::from(view.constant_slot),
            view.constant_dword,
            view.byte_offset,
            u32::from(view.shift),
        ] {
            word(&mut out, value);
        }
    }
    word(&mut out, u32::from(xcl_abi));
    out
}

fn decode(bytes: &[u8]) -> Option<(HostShader, bool)> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_indexed_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_indexed_views.push(victory_shader::resource::RootIndexedView {
            table_slot: u8::try_from(r.word()?).ok()?,
            constant_slot: u8::try_from(r.word()?).ok()?,
            constant_dword: r.word()?,
            byte_offset: r.word()?,
            shift: u8::try_from(r.word()?).ok()?,
        });
    }
    let xcl_abi = r.word()? != 0;
    r.0.is_empty().then_some((
        HostShader {
            stage,
            dxil,
            user_data,
            samplers,
            static_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            root_indexed_views,
        },
        xcl_abi,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok((shader(), false))
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok((shader(), false)));
        assert_eq!(cached(&cache, &build, container, translate), Ok((shader(), false)));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok((shader(), false)));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH2 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok((shader(), false)));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok((shader(), false)));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<(HostShader, bool), _>("unsupported".to_owned());
        assert!(cached(&cache, &build, b"other", fail).is_err());
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader(), true);
        assert_eq!(decode(&bytes), Some((shader(), true)));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }
}
