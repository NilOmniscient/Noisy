// Simple perlin noise library.
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

pub struct Perlin {
    size: usize,
    mask: usize,
    seed: usize,
    octaves: usize,
    frequency: f64,
    amplitude: f64,
    lacunarity: f64,
    persistence: f64,
    warp_size: f64,
    warp_strength: f64,
    erosion_factor: f64,
    seamless: bool,
    domain_warping: bool,
    erosion: bool,
}

impl Default for Perlin {
    fn default() -> Perlin {
        Perlin {
            size: 256,
            mask: 255,
            seed: 1337,
            octaves: 5,
            frequency: 0.05,
            amplitude: 1.0,
            lacunarity: 2.0,
            persistence: 0.5,
            warp_size: 0.35,
            warp_strength: 0.20,
            erosion_factor: 0.35,
            seamless: false,
            domain_warping: false,
            erosion: false,
        }
    }
}

impl Perlin {
    pub fn set_size(&mut self, value: usize) {
        let mask = value - 1;
        self.size = value;
        self.mask = mask;
    }
}

struct Tables {
    px: Vec<usize>,
    py: Vec<usize>,
    gx: Vec<f64>,
    gy: Vec<f64>,
}

// This implementation works in flattened 2D -> 1D array representation
pub fn get_noise_map(perlin: Perlin) -> Vec<f64> {
    let width = perlin.size;
    let mut frequency = perlin.frequency;
    if perlin.seamless {
        // Tweak the frequency ever so slightly so that seamless looks properly seamless
        frequency = (1.0 / perlin.size as f64) * (perlin.frequency * perlin.size as f64).round();
        // I don't believe I've ever seen anyone talk about this needing to be done before?
    }
    let mut map: Vec<f64> = vec![0f64; width * width];
    map.par_iter_mut().enumerate().for_each(|(i, m)| {
        let y: usize = i % width;
        let x: usize = (i - y) / width;
        *m = get_noise2d(x as f64, y as f64, frequency, &perlin);
    });

    // Normalize the map values before returning.
    let maxv = *map.par_iter().max_by(|x, y| x.total_cmp(y)).unwrap();
    let minv = *map.par_iter().min_by(|x, y| x.total_cmp(y)).unwrap();
    let denom = maxv - minv;
    map.par_iter_mut().for_each(|m| {
        *m = (*m - minv) / denom;
    });
    return map;
}

fn get_noise2d(x: f64, y: f64, frequency: f64, perlin: &Perlin) -> f64 {
    // Create permutation and gradient tables dynamically.
    let tables = generate_tables(perlin.seed, perlin.size);

    let mut f = frequency;
    let mut a = perlin.amplitude;

    let mut ret = 0f64;
    let mut dx = 0f64;
    let mut dy = 0f64;

    // Basic Fractal Brownian Motion process.
    for _ in 0..perlin.octaves {
        // Period is used for seamless noise.
        let period: usize = (perlin.size as f64 * f).floor() as usize;

        let noise = noise2d(
            f * x,
            f * y,
            period,
            perlin.mask as usize,
            &tables,
            perlin.seamless,
            perlin.erosion,
        );
        let mut r1 = noise[0];

        // This is based on some of Inigo Quilez' articles I think.
        if perlin.domain_warping {
            let mut tnoise = noise2d(
                (x + perlin.warp_size) * f,
                (y + perlin.warp_size) * f,
                period,
                perlin.mask as usize,
                &tables,
                perlin.seamless,
                perlin.erosion,
            );
            let r2: f64 = tnoise[0];
            tnoise = noise2d(
                (x + perlin.warp_strength * r1) * f,
                (y + perlin.warp_strength * r2) * f,
                period,
                perlin.mask as usize,
                &tables,
                perlin.seamless,
                perlin.erosion,
            );
            r1 = tnoise[0];
        }
        r1 = a * r1;

        // Also based on Inigo Quilez work.
        if perlin.erosion {
            dx += noise[1];
            dy += noise[2];
            r1 /= 1f64 + perlin.erosion_factor * (dx * dx + dy * dy);
        }
        ret += r1;
        a *= perlin.persistence;
        f *= perlin.lacunarity;
    }
    return ret;
}
fn noise2d(
    x: f64,
    y: f64,
    period: usize,
    mask: usize,
    tables: &Tables,
    seamless: bool,
    simulate_erosion: bool,
) -> [f64; 3] {
    let mut x0: usize = x.floor() as usize;
    let mut y0: usize = y.floor() as usize;

    let fx0: f64 = x - x0 as f64;
    let fy0: f64 = y - y0 as f64;
    let fx1: f64 = fx0 - 1.0;
    let fy1: f64 = fy0 - 1.0;

    let mut x1 = x0 + 1;
    let mut y1 = y0 + 1;
    if seamless {
        x0 %= period;
        x1 %= period;
        y0 %= period;
        y1 %= period;
    }
    x0 &= mask;
    x1 &= mask;
    y0 &= mask;
    y1 &= mask;

    let u: f64 = fx0 * fx0 * fx0 * (fx0 * (fx0 * 6f64 - 15f64) + 10f64);
    let v: f64 = fy0 * fy0 * fy0 * (fy0 * (fy0 * 6f64 - 15f64) + 10f64);
    let uv: f64 = u * v;

    // Use two permutation tables instead of a nested lookup on one.
    let h00: usize = (tables.px[x0 as usize] ^ tables.py[y0 as usize]) as usize;
    let h10: usize = (tables.px[x1 as usize] ^ tables.py[y0 as usize]) as usize;
    let h01: usize = (tables.px[x0 as usize] ^ tables.py[y1 as usize]) as usize;
    let h11: usize = (tables.px[x1 as usize] ^ tables.py[y1 as usize]) as usize;

    // Same for gradient tables.
    let a: f64 = tables.gx[h00] as f64 * fx0 + tables.gy[h00] as f64 * fy0;
    let b: f64 = tables.gx[h10] as f64 * fx1 + tables.gy[h10] as f64 * fy0;
    let c: f64 = tables.gx[h01] as f64 * fx0 + tables.gy[h01] as f64 * fy1;
    let d: f64 = tables.gx[h11] as f64 * fx1 + tables.gy[h11] as f64 * fy1;

    let k0: f64 = 1.0 - u - v + uv;
    let k1: f64 = u - uv;
    let k2: f64 = v - uv;
    let k3: f64 = uv;

    let uv: f64 = a * k0 + b * k1 + c * k2 + d * k3;

    // And now the derivatives.
    let mut dx: f64 = 0.0;
    let mut dy: f64 = 0.0;
    if simulate_erosion {
        // If we aren't simulating erosion, just skip, save some calcs.
        let du: f64 = 30.0 * fx0 * fx0 * (fx0 * (fx0 - 2.0) + 1.0);
        let dv: f64 = 30.0 * fy0 * fy0 * (fy0 * (fy0 - 2.0) + 1.0);
        let dcon: f64 = a - b - c + d;
        dx = du * (b - a + dcon * v);
        dy = dv * (c - a + dcon * u);
    }
    let out = [uv, dx, dy];
    return out;
}

/*
    This set generates dual gradient tables, and dual permutation tables.
    It's based on the first of three improvements found in
    a technical report by Andrew Kensler et al called "Better Gradient Noise"
    At the moment it can be found here. https://sci.utah.edu/publications/SCITechReports/UUSCI-2008-001.pdf
    End result is an increase in required memory, with a reduction in complexity. 2^(n+1)-2 -> 2N lookups.
*/
fn generate_tables(seed: usize, size: usize) -> Tables {
    let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
    let mut tables: Tables = Tables {
        px: (0..size).collect(),
        py: (0..size).collect(),
        gx: Vec::with_capacity(size),
        gy: Vec::with_capacity(size),
    };

    tables.px.shuffle(&mut rng);
    tables.py.shuffle(&mut rng);

    for _ in 0..size {
        let theta = rng.random::<f64>() * std::f64::consts::TAU;
        tables.gx.push(theta.cos());
        tables.gy.push(theta.cos());
    }

    return tables;
}
