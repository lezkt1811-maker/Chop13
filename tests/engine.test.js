// Run with: node tests/engine.test.js
// Uses synthetic signals only (no copyrighted audio).
const assert = require('assert');
const E = require('../js/engine.js');

const SR = 44100;

// Kick-like click every beat at `bpm`, optional stereo offset and tone bed.
function clickTrack({ bpm = 120, seconds = 12, stereoWidth = 0, bed = 0 } = {}) {
    const n = Math.floor(seconds * SR);
    const L = new Float32Array(n), R = new Float32Array(n);
    const beat = Math.round(SR * 60 / bpm);
    for (let i = 0; i < n; i++) {
        const t = i / SR;
        let v = bed * Math.sin(2 * Math.PI * 220 * t);
        const pos = i % beat;
        if (pos < SR * 0.02) {
            const env = Math.exp(-pos / (SR * 0.006));
            v += 0.8 * env * Math.sin(2 * Math.PI * 60 * t);
        }
        L[i] = v;
        R[i] = stereoWidth ? v * (1 - stereoWidth) + (Math.sin(2 * Math.PI * 1500 * t) * 0.3 * stereoWidth) : v;
    }
    return [L, R];
}

const tests = [];
const test = (name, fn) => tests.push({ name, fn });

test('BPM estimate lands near 120 for a 120 BPM click track', () => {
    const a = E.analyzeSong(clickTrack({ bpm: 120 }), SR);
    assert.ok(a.bpm, 'bpm should be estimated');
    assert.ok(Math.abs(a.bpm - 120) < 3, 'expected ~120, got ' + a.bpm);
});

test('BPM estimate lands near 90 for a 90 BPM click track', () => {
    const a = E.analyzeSong(clickTrack({ bpm: 90, seconds: 16 }), SR);
    assert.ok(Math.abs(a.bpm - 90) < 3 || Math.abs(a.bpm - 180) < 3, 'expected ~90 (or octave), got ' + a.bpm);
});

test('mono input reports zero stereo width', () => {
    const [L] = clickTrack({ bpm: 100 });
    const a = E.analyzeSong([L, L], SR);
    assert.strictEqual(a.stereo.width, 0);
    assert.ok(Math.abs(a.stereo.correlation - 1) < 0.01);
});

test('wide stereo input reports positive width', () => {
    const a = E.analyzeSong(clickTrack({ bpm: 100, stereoWidth: 1 }), SR);
    assert.ok(a.stereo.width > 0.3, 'width ' + a.stereo.width);
});

test('silence is detected', () => {
    const z = new Float32Array(SR * 2);
    const a = E.analyzeSong([z, z], SR);
    assert.ok(a.silenceRatio > 0.99, 'silence ' + a.silenceRatio);
    assert.strictEqual(a.bpm, null);
});

test('analysis does not mutate input channels', () => {
    const ch = clickTrack({ bpm: 110 });
    const before = ch.map(c => Float32Array.from(c));
    E.analyzeSong(ch, SR);
    ch.forEach((c, i) => assert.deepStrictEqual(c, before[i]));
});

test('every profile produces in-range parameters', () => {
    const a = E.analyzeSong(clickTrack({ bpm: 96, bed: 0.2 }), SR);
    for (const id of E.PROFILE_ORDER) {
        const r = E.recommend(a, id).params;
        assert.ok(r.speed >= 0.70 && r.speed <= 0.95, id + ' speed ' + r.speed);
        assert.ok(r.bass >= 0 && r.bass <= 12, id + ' bass');
        assert.ok(r.filter >= 3000 && r.filter <= 14000, id + ' filter');
        assert.ok(r.width >= 0.4 && r.width <= 0.65, id + ' width ' + r.width);
        assert.ok(r.limiter === -1, id + ' limiter');
        assert.ok(r.chopMs >= 40 && r.chopMs <= 600, id + ' chop');
    }
});

test('profiles produce genuinely different settings', () => {
    const a = E.analyzeSong(clickTrack({ bpm: 96, bed: 0.2 }), SR);
    const sigs = E.PROFILE_ORDER.map(id => JSON.stringify(E.recommend(a, id).params));
    assert.strictEqual(new Set(sigs).size, E.PROFILE_ORDER.length);
});

test('faster tempo gets more slowdown than slower tempo (same profile)', () => {
    const slow = E.recommend({ ...E.analyzeSong(clickTrack({ bpm: 80 }), SR) }, 'h-town').params.speed;
    const fast = E.recommend({ ...E.analyzeSong(clickTrack({ bpm: 130 }), SR) }, 'h-town').params.speed;
    assert.ok(fast <= slow, `fast ${fast} should be <= slow ${slow}`);
});

test('vocal preservation keeps speed high and filter open', () => {
    const a = E.analyzeSong(clickTrack({ bpm: 100 }), SR);
    const r = E.recommend(a, 'vocal-preservation').params;
    assert.ok(r.speed >= 0.86, 'speed ' + r.speed);
    assert.ok(r.filter >= 8000, 'filter ' + r.filter);
});

test('midnight ride has more bass than dream sequence on the same track', () => {
    const a = E.analyzeSong(clickTrack({ bpm: 100 }), SR);
    assert.ok(E.recommend(a, 'midnight-ride').params.bass > E.recommend(a, 'dream-sequence').params.bass);
});

test('pickProfile returns all profiles ranked', () => {
    const a = E.analyzeSong(clickTrack({ bpm: 100 }), SR);
    const rank = E.pickProfile(a);
    assert.strictEqual(rank.length, E.PROFILE_ORDER.length);
    assert.ok(rank[0].score >= rank[1].score);
});

test('surprise stays inside safe limits and varies between calls', () => {
    const a = E.analyzeSong(clickTrack({ bpm: 100, bed: 0.2 }), SR);
    let seed = 1;
    const rng = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const seen = new Set();
    for (let i = 0; i < 30; i++) {
        const p = E.surprise(a, 'deep-soul', rng).params;
        assert.ok(p.speed >= 0.72 && p.speed <= 0.92);
        assert.ok(!(p.filter < 4500 && p.bass > 10), 'mud rule');
        assert.ok(p.width <= 0.65);
        seen.add(JSON.stringify(p));
    }
    assert.ok(seen.size > 10, 'expected variety, got ' + seen.size);
});

test('fftInPlace matches a known single-bin tone', () => {
    const N = 64, re = new Float64Array(N), im = new Float64Array(N);
    for (let i = 0; i < N; i++) re[i] = Math.cos(2 * Math.PI * 4 * i / N);
    E.fftInPlace(re, im);
    const mag = (k) => Math.hypot(re[k], im[k]);
    assert.ok(Math.abs(mag(4) - N / 2) < 1e-6, 'bin 4 magnitude ' + mag(4));
    assert.ok(mag(10) < 1e-6);
});

let failed = 0;
for (const t of tests) {
    try { t.fn(); console.log('ok   ' + t.name); }
    catch (e) { failed++; console.log('FAIL ' + t.name + '\n     ' + e.message); }
}
console.log(`\n${tests.length - failed}/${tests.length} passed`);
process.exit(failed ? 1 : 0);
