/*
 * Chop13 Song Engine (pure logic, no DOM, no Tone.js)
 *
 *  - analyzeSong(channels, sampleRate): measures a decoded track.
 *  - recommend(analysis, profileId): turns a measurement into one set of
 *    processing parameters plus plain-English reasons.
 *  - pickProfile(analysis): ranks the listening profiles for a track.
 *  - surprise(analysis, profileId, rng): a random but safe variation.
 *
 * Everything here is deterministic except surprise(), and it only reads its inputs.
 * Measurements are labelled as measured or estimated in analysis.reliability.
 */
(function (root) {
    'use strict';

    const FFT_SIZE = 4096;          // ~10.8 Hz bins at 44.1 kHz, good enough for sub-bass
    const SPECTRUM_FRAMES = 300;    // evenly spaced analysis frames across the track
    const BLOCK_SEC = 0.05;         // 50 ms loudness blocks
    const SILENCE_DB = -50;

    const BANDS = [
        { id: 'sub',      label: 'Sub',      lo: 20,   hi: 60 },
        { id: 'bass',     label: 'Bass',     lo: 60,   hi: 250 },
        { id: 'lowMid',   label: 'Low-mid',  lo: 250,  hi: 500 },
        { id: 'mid',      label: 'Mid',      lo: 500,  hi: 2000 },
        { id: 'presence', label: 'Presence', lo: 2000, hi: 6000 },
        { id: 'air',      label: 'Air',      lo: 6000, hi: 16000 }
    ];
    const VOCAL_LO = 300, VOCAL_HI = 3400;

    // ---------- Listening profiles: each one is a different processing recipe ----------
    // speed: slowdown before tempo adjustment. bass/mid in dB. filter in Hz.
    // reverbWet 0..1, reverbDecay seconds, delayWet 0..1, delayBeats in beats,
    // chopBeats: stutter slice length in beats. width: stereo widener (0.5 = unchanged, max 0.65).
    const PROFILES = {
        'h-town': {
            id: 'h-town', name: 'H-Town Classic',
            tagline: 'Houston syrup: slow, heavy low end, tight chops',
            base: { speed: 0.82, bass: 8, mid: 0, hpf: 30, filter: 9000, reverbWet: 0.22, reverbDecay: 3.0,
                    delayWet: 0, delayBeats: 0.75, delayFeedback: 0.25, compRatio: 3, saturation: 0.04,
                    width: 0.5, chopBeats: 0.25 }
        },
        'deep-soul': {
            id: 'deep-soul', name: 'Deep Soul',
            tagline: 'Warm, dark and emotional, with the vocal kept forward',
            base: { speed: 0.80, bass: 5, mid: 2, hpf: 30, filter: 5200, reverbWet: 0.30, reverbDecay: 4.2,
                    delayWet: 0.06, delayBeats: 0.75, delayFeedback: 0.3, compRatio: 2.5, saturation: 0.12,
                    width: 0.5, chopBeats: 0.5 }
        },
        'velvet-underground': {
            id: 'velvet-underground', name: 'Velvet Underground',
            tagline: 'Atmospheric, spacious and smooth',
            base: { speed: 0.84, bass: 4, mid: -0.5, hpf: 28, filter: 8000, reverbWet: 0.42, reverbDecay: 5.5,
                    delayWet: 0.12, delayBeats: 0.5, delayFeedback: 0.3, compRatio: 2, saturation: 0.02,
                    width: 0.62, chopBeats: 0.5 }
        },
        'midnight-ride': {
            id: 'midnight-ride', name: 'Midnight Ride',
            tagline: 'Deep bass, hypnotic rhythm and a subtle echo',
            base: { speed: 0.78, bass: 11, mid: -2, hpf: 26, filter: 3800, reverbWet: 0.16, reverbDecay: 2.4,
                    delayWet: 0.22, delayBeats: 0.75, delayFeedback: 0.35, compRatio: 4, saturation: 0.06,
                    width: 0.45, chopBeats: 0.25 }
        },
        'dream-sequence': {
            id: 'dream-sequence', name: 'Dream Sequence',
            tagline: 'Dreamy reverb and hazy, smeared textures',
            base: { speed: 0.80, bass: 4, mid: -1, hpf: 30, filter: 4200, reverbWet: 0.55, reverbDecay: 6.0,
                    delayWet: 0.2, delayBeats: 1, delayFeedback: 0.4, compRatio: 2, saturation: 0.05,
                    width: 0.6, chopBeats: 1 }
        },
        'vocal-preservation': {
            id: 'vocal-preservation', name: 'Vocal Preservation',
            tagline: 'Slower but clear: the vocal stays intact and sharp',
            base: { speed: 0.89, bass: 3, mid: 2.5, hpf: 30, filter: 11000, reverbWet: 0.10, reverbDecay: 2.0,
                    delayWet: 0, delayBeats: 0.5, delayFeedback: 0.2, compRatio: 2, saturation: 0.02,
                    width: 0.5, chopBeats: 0.5 }
        }
    };
    const PROFILE_ORDER = Object.keys(PROFILES);

    // ---------- Helpers ----------
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    const round = (v, step) => Math.round(v / step) * step;
    const toDb = (x) => 10 * Math.log10(Math.max(x, 1e-12));
    const toDb20 = (x) => 20 * Math.log10(Math.max(x, 1e-12));

    function percentile(sorted, p) {
        if (!sorted.length) return 0;
        const idx = clamp(p * (sorted.length - 1), 0, sorted.length - 1);
        const lo = Math.floor(idx), hi = Math.ceil(idx);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
    }

    // In-place iterative radix-2 FFT.
    function fftInPlace(re, im) {
        const n = re.length;
        for (let i = 1, j = 0; i < n; i++) {
            let bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                let t = re[i]; re[i] = re[j]; re[j] = t;
                t = im[i]; im[i] = im[j]; im[j] = t;
            }
        }
        for (let len = 2; len <= n; len <<= 1) {
            const ang = -2 * Math.PI / len;
            const wRe = Math.cos(ang), wIm = Math.sin(ang);
            const half = len >> 1;
            for (let i = 0; i < n; i += len) {
                let cRe = 1, cIm = 0;
                for (let k = 0; k < half; k++) {
                    const a = i + k, b = a + half;
                    const bRe = re[b] * cRe - im[b] * cIm;
                    const bIm = re[b] * cIm + im[b] * cRe;
                    re[b] = re[a] - bRe; im[b] = im[a] - bIm;
                    re[a] += bRe; im[a] += bIm;
                    const nRe = cRe * wRe - cIm * wIm;
                    cIm = cRe * wIm + cIm * wRe;
                    cRe = nRe;
                }
            }
        }
    }

    // ---------- Analysis ----------
    function analyzeSong(channels, sampleRate) {
        const chs = Array.from(channels).filter(Boolean);
        if (!chs.length || !chs[0].length) throw new Error('Empty audio');
        const sr = sampleRate;
        const L = chs[0];
        const R = chs.length > 1 ? chs[1] : chs[0];
        const n = L.length;
        const duration = n / sr;
        const stereo = chs.length > 1;

        // Full-scan pass: peak, power, stereo relations, and a mono mix.
        const mono = new Float32Array(n);
        let peak = 0, sumSqMono = 0, sLL = 0, sRR = 0, sLR = 0, sMM = 0, sSS = 0;
        for (let i = 0; i < n; i++) {
            const l = L[i], r = R[i];
            const m = (l + r) * 0.5, s = (l - r) * 0.5;
            mono[i] = m;
            const a = Math.max(Math.abs(l), Math.abs(r));
            if (a > peak) peak = a;
            sumSqMono += m * m;
            sLL += l * l; sRR += r * r; sLR += l * r;
            sMM += m * m; sSS += s * s;
        }
        const rms = Math.sqrt(sumSqMono / n);
        const peakDb = toDb20(peak);
        const rmsDb = toDb20(rms);

        // 50 ms blocks: silence, dynamic range, and transient density.
        const blockLen = Math.max(1, Math.round(sr * BLOCK_SEC));
        const blockDb = [];
        for (let start = 0; start + blockLen <= n; start += blockLen) {
            let acc = 0;
            for (let i = start; i < start + blockLen; i++) acc += mono[i] * mono[i];
            blockDb.push(toDb(acc / blockLen));
        }
        const silent = blockDb.filter(d => d < SILENCE_DB).length;
        const audible = blockDb.filter(d => d >= SILENCE_DB).sort((x, y) => x - y);
        const dynamicRangeDb = audible.length ? percentile(audible, 0.95) - percentile(audible, 0.10) : 0;
        let onsets = 0;
        for (let i = 1; i < blockDb.length; i++) {
            if (blockDb[i] >= SILENCE_DB && blockDb[i] - blockDb[i - 1] > 6) onsets++;
        }
        const transientDensity = onsets / Math.max(duration, 1e-6);   // onsets per second

        // Spectrum: average band power across evenly spaced Hann-windowed frames.
        const bandPow = {};
        BANDS.forEach(b => bandPow[b.id] = 0);
        let vocalPow = 0, totalPow = 0, centroidNum = 0, centroidDen = 0;
        const binHz = sr / FFT_SIZE;
        const re = new Float64Array(FFT_SIZE), im = new Float64Array(FFT_SIZE);
        const win = new Float64Array(FFT_SIZE);
        for (let i = 0; i < FFT_SIZE; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FFT_SIZE - 1));
        const frames = n >= FFT_SIZE ? SPECTRUM_FRAMES : 1;
        const maxStart = Math.max(0, n - FFT_SIZE);
        for (let f = 0; f < frames; f++) {
            const start = frames === 1 ? 0 : Math.floor(f * maxStart / (frames - 1));
            for (let i = 0; i < FFT_SIZE; i++) {
                re[i] = (start + i < n ? mono[start + i] : 0) * win[i];
                im[i] = 0;
            }
            fftInPlace(re, im);
            for (let k = 1; k < FFT_SIZE / 2; k++) {
                const p = re[k] * re[k] + im[k] * im[k];
                const hz = k * binHz;
                totalPow += p;
                centroidNum += hz * p;
                centroidDen += p;
                if (hz >= VOCAL_LO && hz <= VOCAL_HI) vocalPow += p;
                for (const b of BANDS) {
                    if (hz >= b.lo && hz < b.hi) { bandPow[b.id] += p; break; }
                }
            }
        }
        const bandsPct = {}, bandsDb = {};
        BANDS.forEach(b => {
            bandsPct[b.id] = totalPow > 0 ? bandPow[b.id] / totalPow : 0;
            bandsDb[b.id] = toDb(bandPow[b.id] / Math.max(frames, 1));
        });
        const vocalRatio = totalPow > 0 ? vocalPow / totalPow : 0;
        const centroidHz = centroidDen > 0 ? centroidNum / centroidDen : 0;

        // Stereo: correlation, side-to-mid energy, and channel balance.
        const correlation = (sLL > 0 && sRR > 0) ? clamp(sLR / Math.sqrt(sLL * sRR), -1, 1) : 1;
        const stereoAnalysis = {
            isStereo: stereo,
            correlation: round(correlation, 0.01),
            width: stereo ? round(clamp(1 - correlation, 0, 1), 0.01) : 0,
            sideToMidDb: round(toDb(sSS / Math.max(sMM, 1e-12)), 0.1),
            balanceDb: (sLL > 0 && sRR > 0) ? round(toDb(sLL / sRR) / 2, 0.1) : 0 // + means left louder
        };

        // Tempo: autocorrelation of an onset-strength envelope (estimate).
        const bpmInfo = estimateBpm(mono, sr);

        const analysis = {
            duration: round(duration, 0.01),
            sampleRate: sr,
            channels: chs.length,
            peakDb: round(peakDb, 0.1),
            rmsDb: round(rmsDb, 0.1),
            crestDb: round(peakDb - rmsDb, 0.1),
            dynamicRangeDb: round(dynamicRangeDb, 0.1),
            silenceRatio: round(blockDb.length ? silent / blockDb.length : 0, 0.01),
            transientDensity: round(transientDensity, 0.01),
            bpm: bpmInfo.bpm,
            bpmConfidence: bpmInfo.confidence,
            bandsPct: bandsPct,
            bandsDb: bandsDb,
            centroidHz: Math.round(centroidHz),
            vocal: { ratio: round(vocalRatio, 0.01) },
            stereo: stereoAnalysis,
            reliability: {
                loudness: 'measured',
                spectrum: 'measured (FFT, sampled frames)',
                stereo: 'measured',
                dynamics: 'measured (50 ms blocks)',
                bpm: bpmInfo.bpm ? 'estimated (onset autocorrelation, can be off by a multiple)' : 'not enough audio to estimate',
                vocals: 'estimated (share of energy in 300-3400 Hz, not vocal detection)'
            }
        };
        return analysis;
    }

    // Estimated tempo from an onset envelope. Returns { bpm, confidence }.
    function estimateBpm(mono, sr) {
        const hop = 512, win = 1024;
        const count = Math.floor((mono.length - win) / hop) + 1;
        if (count < 16 || mono.length / sr < 4) return { bpm: null, confidence: 0 };
        const energy = new Float64Array(count);
        for (let f = 0; f < count; f++) {
            let acc = 0;
            const s = f * hop;
            for (let i = 0; i < win; i++) acc += mono[s + i] * mono[s + i];
            energy[f] = acc / win;
        }
        const env = new Float64Array(count);
        for (let f = 1; f < count; f++) {
            env[f] = Math.max(0, Math.log(energy[f] + 1e-9) - Math.log(energy[f - 1] + 1e-9));
        }
        let mean = 0;
        for (let f = 0; f < count; f++) mean += env[f];
        mean /= count;
        for (let f = 0; f < count; f++) env[f] -= mean;
        const fps = sr / hop;
        const ac = (lag) => {
            let s = 0;
            for (let f = 0; f + lag < count; f++) s += env[f] * env[f + lag];
            return s / (count - lag);
        };
        const ac0 = ac(0) || 1e-9;
        const minLag = Math.max(2, Math.floor(60 * fps / 200));
        const maxLag = Math.min(count - 2, Math.ceil(60 * fps / 60));
        let bestLag = -1, bestVal = -Infinity;
        const vals = {};
        for (let lag = minLag; lag <= maxLag; lag++) {
            vals[lag] = ac(lag) / ac0;
            if (vals[lag] > bestVal) { bestVal = vals[lag]; bestLag = lag; }
        }
        if (bestLag < 0) return { bpm: null, confidence: 0 };
        // Parabolic refinement for sub-frame accuracy.
        const y1 = vals[bestLag - 1] ?? bestVal, y2 = bestVal, y3 = vals[bestLag + 1] ?? bestVal;
        const denom = y1 - 2 * y2 + y3;
        const delta = denom !== 0 ? 0.5 * (y1 - y3) / denom : 0;
        let bpm = 60 * fps / (bestLag + clamp(delta, -0.5, 0.5));
        // Fold into a musical range (70-180 BPM).
        while (bpm < 70) bpm *= 2;
        while (bpm > 180) bpm /= 2;
        return { bpm: Math.round(bpm * 10) / 10, confidence: round(clamp(bestVal, 0, 1), 0.01) };
    }

    // ---------- Profile ranking ----------
    function pickProfile(a) {
        const b = a.bandsPct;
        const bass = b.sub + b.bass;
        const bpm = a.bpm || 100;
        const scores = {};
        const why = {};
        PROFILE_ORDER.forEach(id => { scores[id] = 0; why[id] = []; });
        const add = (id, pts, text) => { scores[id] += pts; why[id].push(text); };

        if (a.centroidHz < 1800) add('deep-soul', 2, 'dark spectrum (centroid ' + a.centroidHz + ' Hz)');
        if (a.vocal.ratio > 0.4) add('deep-soul', 1, 'strong midrange vocal energy');
        if (a.vocal.ratio > 0.4) add('vocal-preservation', 2, 'strong vocal-band energy');
        if (bpm < 95) add('vocal-preservation', 1, 'slower tempo leaves room for the vocal');
        if (a.stereo.width > 0.35) add('velvet-underground', 2, 'wide stereo image');
        if (a.dynamicRangeDb > 12) add('velvet-underground', 1, 'wide dynamic range suits smooth space');
        if (bass > 0.35 && bpm >= 90) add('midnight-ride', 2, 'heavy bass with a driving tempo');
        if (a.transientDensity > 4) add('midnight-ride', 1, 'dense rhythmic hits');
        if (a.silenceRatio > 0.05 || a.transientDensity < 1.5) add('dream-sequence', 2, 'sparse, open arrangement');
        if (a.centroidHz < 2500) add('dream-sequence', 1, 'soft top end');
        if (bpm >= 80 && bpm <= 105) add('h-town', 2, 'classic slow-jam tempo');
        if (bass > 0.25) add('h-town', 1, 'solid bass presence');
        add('h-town', 0.5, 'the default chopped-and-screwed sound');

        const ranking = PROFILE_ORDER
            .map(id => ({ id, score: scores[id], reasons: why[id] }))
            .sort((x, y) => y.score - x.score || PROFILE_ORDER.indexOf(x.id) - PROFILE_ORDER.indexOf(y.id));
        return ranking;
    }

    // ---------- Recommendation ----------
    // Returns { profileId, profileName, params, reasons, pitchSemitones }
    function recommend(a, profileId) {
        const p = PROFILES[profileId];
        if (!p) throw new Error('Unknown profile: ' + profileId);
        const b = p.base;
        const bpm = a.bpm || 100;
        const bpmN = clamp((bpm - 70) / 110, 0, 1);       // 0 = slow, 1 = fast
        const bp = a.bandsPct;
        const bassFrac = bp.sub + bp.bass;
        const muddy = bp.lowMid > 0.2;
        const harsh = (bp.presence + bp.air) > 0.32;
        const dark = a.centroidHz < 1800;
        const sparse = a.transientDensity < 1.5;
        const reasons = [];

        // Speed: fast tracks get a little more slowdown, slow ones a little less.
        let speed = b.speed + (0.5 - bpmN) * 0.06;
        if (profileId === 'vocal-preservation') speed = clamp(speed, 0.86, 0.92);
        speed = clamp(round(speed, 0.01), 0.70, 0.95);
        const pitchSemitones = round(12 * Math.log2(speed), 0.1);
        reasons.push(`Speed ${speed.toFixed(2)}x (about ${pitchSemitones.toFixed(1)} semitones): ` +
            (bpm > 110 ? 'a fast track gets slightly more slowdown for weight.'
                : bpm < 90 ? 'a slow track gets a lighter slowdown to keep its groove.'
                : 'standard slowdown for this tempo.'));

        // Bass: boost what is light, trim what is already heavy, and cap it when the low-mids are muddy.
        let bass = b.bass + (bassFrac < 0.25 ? 2 : bassFrac > 0.45 ? -2.5 : 0);
        if (muddy) bass = Math.min(bass, 6);
        bass = clamp(round(bass, 0.5), 0, 12);
        reasons.push(`Bass +${bass} dB: ` + (bassFrac > 0.45 ? 'the track is already bass-heavy, so the boost is reduced.'
            : muddy ? 'low-mids are crowded, so the boost is capped to avoid mud.'
            : 'the low end is light, so it gets extra weight.'));

        // Mid EQ: keep the vocal forward when the track has little midrange.
        let mid = b.mid + (muddy ? -1.5 : 0) + (a.vocal.ratio < 0.3 ? 1 : 0);
        mid = clamp(round(mid, 0.5), -6, 4);
        reasons.push(`Mid ${mid >= 0 ? '+' : ''}${mid} dB: ` + (muddy ? 'trimmed the crowded low-mids.'
            : a.vocal.ratio < 0.3 ? 'lifted the midrange to protect the vocal.'
            : 'kept near neutral.'));

        // High-pass: remove subsonic rumble when sub energy is high.
        let hpf = b.hpf + (bp.sub > 0.10 ? 8 : 0);
        hpf = clamp(Math.round(hpf), 20, 45);
        reasons.push(`High-pass ${hpf} Hz: ` + (bp.sub > 0.10 ? 'the sub band is large, so rumble is cut higher.'
            : 'standard rumble cut.'));

        // Low-pass: tame harsh highs and keep the vocal bright enough for the profile.
        let filter = b.filter * (harsh ? 0.82 : 1) * (dark ? 1.05 : 1);
        if (profileId === 'vocal-preservation') filter = Math.max(filter, 8000);
        filter = clamp(Math.round(filter / 100) * 100, 3000, 14000);
        reasons.push(`Low-pass ${filter} Hz: ` + (harsh ? 'the highs are already bright, so they are rolled off.'
            : 'the top end is not harsh, so the filter stays open.'));

        // Reverb: sparse tracks have room for more; dense ones need less.
        let wet = b.reverbWet + (sparse ? 0.05 : 0) - (a.transientDensity > 4 ? 0.05 : 0);
        wet = clamp(round(wet, 0.05), 0, 0.7);
        let decay = clamp(round(b.reverbDecay + (1 - speed) * 4, 0.1), 1.5, 8);
        reasons.push(`Reverb ${Math.round(wet * 100)}% wet, ${decay}s decay: ` +
            (sparse ? 'sparse arrangement, so the space can be fuller.' : 'busy arrangement, so the space is kept modest.'));

        // Echo: timed to the tempo after slowing.
        const beatSec = 60 / bpm;
        const delayTime = clamp(round(beatSec * b.delayBeats / speed, 0.01), 0.08, 1.2);
        const delayWet = b.delayWet;
        const delayFeedback = b.delayFeedback;
        reasons.push(delayWet > 0
            ? `Echo ${Math.round(delayWet * 100)}% at ${delayTime.toFixed(2)}s, timed to the tempo.`
            : 'No echo for this profile.');

        // Compression: dynamic tracks get more control; already-compressed tracks get less.
        let ratio = b.compRatio + (a.crestDb > 14 ? 0.5 : a.crestDb < 9 ? -0.5 : 0);
        ratio = clamp(round(ratio, 0.5), 1.5, 6);
        const threshold = clamp(Math.round(a.rmsDb + 4), -26, -8);
        reasons.push(`Compression ${ratio}:1 above ${threshold} dB: ` +
            (a.crestDb > 14 ? 'the track is very dynamic, so it is evened out.'
                : a.crestDb < 9 ? 'the track is already compressed, so it is only lightly held.'
                : 'a moderate level control.'));

        // Saturation: a little warmth, more for dark tracks.
        let sat = b.saturation + (dark ? 0.02 : 0);
        sat = clamp(round(sat, 0.01), 0, 0.2);
        reasons.push(`Warmth ${Math.round(sat * 100)}%: ` + (dark ? 'a dark track gets extra warmth.' : 'light harmonic colour.'));

        // Stereo: keep the source's image; never push past the mono-safe limit.
        let width = b.width + (a.stereo.width < 0.15 ? 0.05 : a.stereo.width > 0.5 ? -0.05 : 0);
        width = clamp(round(width, 0.01), 0.4, 0.65);
        reasons.push(`Stereo width ${width.toFixed(2)} (0.50 = unchanged, max 0.65 for mono safety): ` +
            (a.stereo.isStereo ? (a.stereo.width < 0.15 ? 'the source is nearly mono, so a little width is added.'
                : a.stereo.width > 0.5 ? 'the source is already wide, so width is reduced.' : 'kept close to the source.')
                : 'the source is mono.'));

        // Stutter slice length: follows the beat after slowing.
        const chopMs = clamp(Math.round(beatSec * b.chopBeats / speed * 1000), 40, 600);
        reasons.push(`Chop slice ${chopMs} ms: ${b.chopBeats} beat(s) at this tempo and speed.`);

        return {
            profileId,
            profileName: p.name,
            pitchSemitones,
            reasons,
            params: {
                speed, bass, mid, hpf, filter,
                reverbWet: wet, reverbDecay: decay,
                delayWet, delayTime, delayFeedback,
                compRatio: ratio, compThreshold: threshold,
                saturation: sat, width,
                chopMs,
                limiter: -1
            }
        };
    }

    // ---------- Surprise Me 2.0: random, but inside musically safe limits ----------
    function surprise(a, profileId, rng) {
        const rand = rng || Math.random;
        const base = recommend(a, profileId);
        const bp = base.params;
        const u = (lo, hi) => lo + rand() * (hi - lo);
        for (let attempt = 0; attempt < 25; attempt++) {
            const cand = {
                speed: round(clamp(bp.speed + u(-0.03, 0.03), 0.72, 0.92), 0.01),
                bass: round(clamp(bp.bass + u(-2, 2), 0, 12), 0.5),
                mid: round(clamp(bp.mid + u(-1.5, 1.5), -6, 4), 0.5),
                hpf: bp.hpf,
                filter: round(clamp(bp.filter + u(-1200, 1200), 3000, 14000), 100),
                reverbWet: round(clamp(bp.reverbWet + u(-0.1, 0.1), 0, 0.65), 0.05),
                reverbDecay: round(clamp(bp.reverbDecay + u(-1, 1), 1.5, 7), 0.1),
                delayWet: round(clamp(bp.delayWet + u(-0.08, 0.12), 0, 0.3), 0.01),
                delayFeedback: bp.delayFeedback,
                compRatio: round(clamp(bp.compRatio + u(-0.5, 0.5), 1.5, 5), 0.5),
                compThreshold: bp.compThreshold,
                saturation: round(clamp(bp.saturation + u(-0.03, 0.03), 0, 0.15), 0.01),
                width: round(clamp(bp.width + u(-0.08, 0.08), 0.4, 0.65), 0.01),
                limiter: -1
            };
            const beatSec = 60 / (a.bpm || 100);
            const beatChoices = [0.25, 0.5, 1];
            cand.chopMs = clamp(Math.round(beatSec * beatChoices[Math.floor(rand() * 3)] / cand.speed * 1000), 40, 600);
            cand.delayTime = clamp(round(beatSec * (PROFILES[profileId].base.delayBeats) / cand.speed, 0.01), 0.08, 1.2);

            // Safety: avoid mud, overwhelming reverb, and over-wide stereo.
            if (cand.filter < 4500 && cand.bass > 10) continue;
            if (cand.reverbWet > 0.5 && cand.reverbDecay > 6) continue;
            if (cand.delayWet > 0.25 && cand.delayFeedback > 0.35) continue;
            if (cand.mid > 2 && cand.bass > 10) continue;
            return {
                profileId,
                profileName: PROFILES[profileId].name + ' (surprise)',
                pitchSemitones: round(12 * Math.log2(cand.speed), 0.1),
                reasons: ['Randomised within safe limits around the ' + PROFILES[profileId].name + ' recipe for this track.'],
                params: cand
            };
        }
        return base;   // nothing safe found in 25 tries: return the recommendation
    }

    const api = { PROFILES, PROFILE_ORDER, BANDS, analyzeSong, recommend, pickProfile, surprise, fftInPlace };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.SongEngine = api;
})(typeof window !== 'undefined' ? window : globalThis);
