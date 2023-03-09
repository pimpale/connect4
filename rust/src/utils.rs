pub fn crosscorrelate2d<
    const IWIDTH: usize,
    const IHEIGHT: usize,
    const KWIDTH: usize,
    const KHEIGHT: usize,
>(
    input: &[[f32; IWIDTH]; IHEIGHT],
    kernel: &[[f32; KWIDTH]; KHEIGHT],
) -> [[f32; IWIDTH - KWIDTH + 1]; IHEIGHT - KHEIGHT + 1]
{
    let mut output = [[0.0; IWIDTH - KWIDTH + 1]; IHEIGHT - KHEIGHT + 1];
    for y in 0..IHEIGHT - KHEIGHT + 1 {
        for x in 0..IWIDTH - KWIDTH + 1 {
            let mut sum = 0.0;
            for ky in 0..KHEIGHT {
                for kx in 0..KWIDTH {
                    sum = sum + input[y + ky][x + kx] * kernel[ky][kx];
                }
            }
            output[y][x] = sum;
        }
    }
    output
}

pub fn sum<const WIDTH: usize, const HEIGHT: usize>(
    input: &[[f32; WIDTH]; HEIGHT],
) -> f32 {
    let mut sum = 0.0;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            sum = sum + input[y][x];
        }
    }
    sum
}

pub fn cast2d<T, U, const WIDTH: usize, const HEIGHT: usize>(
    input: &[[T; WIDTH]; HEIGHT],
) -> [[U; WIDTH]; HEIGHT]
where
    T: Clone,
    U: From<T> + Default + Copy,
{
    let mut output = [[U::default(); WIDTH]; HEIGHT];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            output[y][x] = input[y][x].clone().into();
        }
    }
    output
}

/// input in [actions,]
pub fn entropy(t:&tch::Tensor, dtype:tch::Kind) -> f32 {
    let epsilon = 1e-8;
    let t = &t.clamp(epsilon, 1.0);
    
    f32::from(-(t * t.log()).sum(dtype))
}

/// produces gumbel noise in the specified shape
pub fn gumbel(shape: &[i64], dtype: tch::Kind) -> tch::Tensor {
    let u = tch::Tensor::rand(shape, (dtype, tch::Device::Cpu));
    let g = -(-u.log()).log();
    g
}