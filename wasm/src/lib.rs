use wasm_bindgen::prelude::*;

/// Decode a PNG/JPEG image from raw bytes into RGBA pixels + dimensions.
/// Returns (width, height, rgba_pixels).
fn decode_image(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    // Minimal PNG decoder — supports 8-bit RGB/RGBA, non-interlaced
    if data.len() >= 8 && &data[0..8] == b"\x89PNG\r\n\x1a\n" {
        return decode_png(data);
    }
    // For JPEG / other formats, we rely on the JS side converting to PNG first
    Err("Unsupported image format — please convert to PNG before uploading".into())
}

fn decode_png(data: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    // We use a tiny inline PNG decoder for WASM (no heavy deps).
    // For production, you'd use the `png` crate, but keeping deps minimal here.
    //
    // Strategy: parse chunks, inflate IDAT, unfilter scanlines.
    let mut pos = 8usize; // skip signature
    let mut width = 0u32;
    let mut height = 0u32;
    #[allow(unused_assignments)]
    let mut bit_depth: u8 = 0;
    let mut color_type: u8 = 0;
    let mut idat_data = Vec::new();

    while pos + 8 <= data.len() {
        let chunk_len = u32::from_be_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        let chunk_type = &data[pos+4..pos+8];
        let chunk_data_start = pos + 8;
        let chunk_data_end = chunk_data_start + chunk_len;
        if chunk_data_end > data.len() { break; }

        match chunk_type {
            b"IHDR" => {
                if chunk_len < 13 { return Err("Invalid IHDR".into()); }
                let d = &data[chunk_data_start..chunk_data_end];
                width = u32::from_be_bytes([d[0], d[1], d[2], d[3]]);
                height = u32::from_be_bytes([d[4], d[5], d[6], d[7]]);
                bit_depth = d[8];
                color_type = d[9];
                if bit_depth != 8 {
                    return Err(format!("Unsupported bit depth: {}", bit_depth));
                }
                if color_type != 2 && color_type != 6 {
                    return Err(format!("Unsupported color type: {} (only RGB/RGBA)", color_type));
                }
            }
            b"IDAT" => {
                idat_data.extend_from_slice(&data[chunk_data_start..chunk_data_end]);
            }
            b"IEND" => break,
            _ => {}
        }
        pos = chunk_data_end + 4; // skip CRC
    }

    if width == 0 || height == 0 {
        return Err("Could not parse PNG dimensions".into());
    }

    // Decompress IDAT (deflate stream wrapped in zlib)
    let raw = inflate_zlib(&idat_data)?;

    let channels: usize = if color_type == 6 { 4 } else { 3 };
    let stride = 1 + width as usize * channels; // filter byte + pixel data
    if raw.len() < stride * height as usize {
        return Err("Decompressed data too short".into());
    }

    // Unfilter scanlines and produce RGBA output
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    let mut prev_row = vec![0u8; width as usize * channels];

    for y in 0..height as usize {
        let row_start = y * stride;
        let filter = raw[row_start];
        let row_data = &raw[row_start + 1..row_start + stride];
        let mut current_row = row_data.to_vec();

        // Apply PNG filter
        match filter {
            0 => {} // None
            1 => { // Sub
                for i in channels..current_row.len() {
                    current_row[i] = current_row[i].wrapping_add(current_row[i - channels]);
                }
            }
            2 => { // Up
                for i in 0..current_row.len() {
                    current_row[i] = current_row[i].wrapping_add(prev_row[i]);
                }
            }
            3 => { // Average
                for i in 0..current_row.len() {
                    let left = if i >= channels { current_row[i - channels] as u16 } else { 0 };
                    let up = prev_row[i] as u16;
                    current_row[i] = current_row[i].wrapping_add(((left + up) / 2) as u8);
                }
            }
            4 => { // Paeth
                for i in 0..current_row.len() {
                    let left = if i >= channels { current_row[i - channels] as i32 } else { 0 };
                    let up = prev_row[i] as i32;
                    let up_left = if i >= channels { prev_row[i - channels] as i32 } else { 0 };
                    current_row[i] = current_row[i].wrapping_add(paeth(left, up, up_left) as u8);
                }
            }
            _ => return Err(format!("Unknown PNG filter: {}", filter)),
        }

        // Write to RGBA output
        for x in 0..width as usize {
            let dst = (y * width as usize + x) * 4;
            if channels == 4 {
                pixels[dst]     = current_row[x * 4];
                pixels[dst + 1] = current_row[x * 4 + 1];
                pixels[dst + 2] = current_row[x * 4 + 2];
                pixels[dst + 3] = current_row[x * 4 + 3];
            } else {
                pixels[dst]     = current_row[x * 3];
                pixels[dst + 1] = current_row[x * 3 + 1];
                pixels[dst + 2] = current_row[x * 3 + 2];
                pixels[dst + 3] = 255;
            }
        }

        prev_row = current_row;
    }

    Ok((width, height, pixels))
}

fn paeth(a: i32, b: i32, c: i32) -> i32 {
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc { a }
    else if pb <= pc { b }
    else { c }
}

/// Minimal zlib/deflate decompressor (enough for PNG IDAT streams).
fn inflate_zlib(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 2 {
        return Err("Zlib data too short".into());
    }
    // Skip 2-byte zlib header
    inflate_deflate(&data[2..])
}

fn inflate_deflate(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut reader = BitReader::new(data);

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;

        match btype {
            0 => {
                // Stored block
                reader.align_byte();
                let len = reader.read_bits(16)? as usize;
                let _nlen = reader.read_bits(16)?;
                for _ in 0..len {
                    output.push(reader.read_bits(8)? as u8);
                }
            }
            1 => inflate_block_fixed(&mut reader, &mut output)?,
            2 => inflate_block_dynamic(&mut reader, &mut output)?,
            _ => return Err("Invalid deflate block type".into()),
        }

        if bfinal != 0 { break; }
    }

    Ok(output)
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    current: u32,
    bits_left: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, current: 0, bits_left: 0 }
    }

    fn read_bits(&mut self, count: u8) -> Result<u32, String> {
        while self.bits_left < count {
            if self.pos >= self.data.len() {
                return Err("Unexpected end of deflate data".into());
            }
            self.current |= (self.data[self.pos] as u32) << self.bits_left;
            self.pos += 1;
            self.bits_left += 8;
        }
        let mask = (1u32 << count) - 1;
        let val = self.current & mask;
        self.current >>= count;
        self.bits_left -= count;
        Ok(val)
    }

    fn align_byte(&mut self) {
        self.current = 0;
        self.bits_left = 0;
    }
}

// Fixed Huffman tables for deflate
fn inflate_block_fixed(reader: &mut BitReader, output: &mut Vec<u8>) -> Result<(), String> {
    loop {
        let sym = decode_fixed_literal(reader)?;
        if sym == 256 { break; }
        if sym < 256 {
            output.push(sym as u8);
        } else {
            let length = decode_length(sym, reader)?;
            let dist_code = read_bits_reversed(reader, 5)?;
            let distance = decode_distance(dist_code, reader)?;
            for _ in 0..length {
                let idx = output.len() - distance;
                output.push(output[idx]);
            }
        }
    }
    Ok(())
}

fn decode_fixed_literal(reader: &mut BitReader) -> Result<u32, String> {
    // Read 7 bits first
    let mut code = read_bits_reversed(reader, 7)?;
    if code <= 0b0010111 { // 256-279
        return Ok(code + 256);
    }
    // Read 1 more bit (8 total)
    let extra = reader.read_bits(1)?;
    code = (code << 1) | extra;
    if code >= 0b00110000 && code <= 0b10111111 { // 0-143
        return Ok(code - 0b00110000);
    }
    if code >= 0b11000000 && code <= 0b11000111 { // 280-287
        return Ok(code - 0b11000000 + 280);
    }
    // Read 1 more bit (9 total)
    let extra2 = reader.read_bits(1)?;
    code = (code << 1) | extra2;
    if code >= 0b110010000 && code <= 0b111111111 { // 144-255
        return Ok(code - 0b110010000 + 144);
    }
    Err(format!("Invalid fixed Huffman code: {}", code))
}

fn read_bits_reversed(reader: &mut BitReader, count: u8) -> Result<u32, String> {
    let bits = reader.read_bits(count)?;
    let mut reversed = 0u32;
    for i in 0..count {
        if bits & (1 << i) != 0 {
            reversed |= 1 << (count - 1 - i);
        }
    }
    Ok(reversed)
}

fn inflate_block_dynamic(reader: &mut BitReader, output: &mut Vec<u8>) -> Result<(), String> {
    let hlit = reader.read_bits(5)? as usize + 257;
    let hdist = reader.read_bits(5)? as usize + 1;
    let hclen = reader.read_bits(4)? as usize + 4;

    // Code length alphabet order
    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
    let mut cl_lengths = [0u8; 19];
    for i in 0..hclen {
        cl_lengths[ORDER[i]] = reader.read_bits(3)? as u8;
    }

    let cl_tree = build_huffman_tree(&cl_lengths)?;

    // Read literal/length + distance code lengths
    let mut lengths = Vec::with_capacity(hlit + hdist);
    while lengths.len() < hlit + hdist {
        let sym = decode_huffman(reader, &cl_tree)?;
        match sym {
            0..=15 => lengths.push(sym as u8),
            16 => {
                let repeat = reader.read_bits(2)? as usize + 3;
                let last = *lengths.last().ok_or("No previous length for repeat")? ;
                for _ in 0..repeat { lengths.push(last); }
            }
            17 => {
                let repeat = reader.read_bits(3)? as usize + 3;
                for _ in 0..repeat { lengths.push(0); }
            }
            18 => {
                let repeat = reader.read_bits(7)? as usize + 11;
                for _ in 0..repeat { lengths.push(0); }
            }
            _ => return Err("Invalid code length symbol".into()),
        }
    }

    let lit_tree = build_huffman_tree(&lengths[..hlit])?;
    let dist_tree = build_huffman_tree(&lengths[hlit..hlit + hdist])?;

    loop {
        let sym = decode_huffman(reader, &lit_tree)?;
        if sym == 256 { break; }
        if sym < 256 {
            output.push(sym as u8);
        } else {
            let length = decode_length(sym, reader)?;
            let dist_sym = decode_huffman(reader, &dist_tree)?;
            let distance = decode_distance(dist_sym, reader)?;
            for _ in 0..length {
                let idx = output.len() - distance;
                output.push(output[idx]);
            }
        }
    }
    Ok(())
}

// Huffman tree as a flat array: each node is (left_child, right_child) or a leaf value
#[derive(Clone)]
enum HuffNode {
    Internal(usize, usize), // left, right indices
    Leaf(u32),
    Empty,
}

struct HuffTree {
    nodes: Vec<HuffNode>,
}

fn build_huffman_tree(lengths: &[u8]) -> Result<HuffTree, String> {
    let max_bits = *lengths.iter().max().unwrap_or(&0) as usize;
    if max_bits == 0 {
        return Ok(HuffTree { nodes: vec![HuffNode::Leaf(0)] });
    }

    // Count codes per length
    let mut bl_count = vec![0u32; max_bits + 1];
    for &l in lengths {
        if l > 0 { bl_count[l as usize] += 1; }
    }

    // Compute starting codes
    let mut next_code = vec![0u32; max_bits + 1];
    let mut code = 0u32;
    for bits in 1..=max_bits {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Build tree
    let mut tree = HuffTree { nodes: vec![HuffNode::Empty] }; // root at index 0

    for (sym, &len) in lengths.iter().enumerate() {
        if len == 0 { continue; }
        let len = len as usize;
        let code = next_code[len];
        next_code[len] += 1;

        let mut node_idx = 0;
        for bit_pos in (0..len).rev() {
            let bit = (code >> bit_pos) & 1;
            let next_idx = match &tree.nodes[node_idx] {
                HuffNode::Internal(left, right) => {
                    if bit == 0 { *left } else { *right }
                }
                HuffNode::Empty => {
                    let left = tree.nodes.len();
                    tree.nodes.push(HuffNode::Empty);
                    let right = tree.nodes.len();
                    tree.nodes.push(HuffNode::Empty);
                    tree.nodes[node_idx] = HuffNode::Internal(left, right);
                    if bit == 0 { left } else { right }
                }
                HuffNode::Leaf(_) => return Err("Huffman tree conflict".into()),
            };
            node_idx = next_idx;
        }
        tree.nodes[node_idx] = HuffNode::Leaf(sym as u32);
    }

    Ok(tree)
}

fn decode_huffman(reader: &mut BitReader, tree: &HuffTree) -> Result<u32, String> {
    let mut node_idx = 0;
    loop {
        match &tree.nodes[node_idx] {
            HuffNode::Leaf(sym) => return Ok(*sym),
            HuffNode::Internal(left, right) => {
                let bit = reader.read_bits(1)?;
                node_idx = if bit == 0 { *left } else { *right };
            }
            HuffNode::Empty => return Err("Invalid Huffman code (empty node)".into()),
        }
    }
}

fn decode_length(sym: u32, reader: &mut BitReader) -> Result<usize, String> {
    const BASE: [usize; 29] = [
        3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258
    ];
    const EXTRA: [u8; 29] = [
        0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0
    ];
    let idx = (sym - 257) as usize;
    if idx >= 29 { return Err("Invalid length code".into()); }
    let extra = reader.read_bits(EXTRA[idx])? as usize;
    Ok(BASE[idx] + extra)
}

fn decode_distance(sym: u32, reader: &mut BitReader) -> Result<usize, String> {
    const BASE: [usize; 30] = [
        1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,
        1025,1537,2049,3073,4097,6145,8193,12289,16385,24577
    ];
    const EXTRA: [u8; 30] = [
        0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13
    ];
    let idx = sym as usize;
    if idx >= 30 { return Err("Invalid distance code".into()); }
    let extra = reader.read_bits(EXTRA[idx])? as usize;
    Ok(BASE[idx] + extra)
}

/// Bilinear upscaling of RGBA pixel data.
fn bilinear_upscale(pixels: &[u8], w: u32, h: u32, scale: u32) -> Vec<u8> {
    let new_w = w * scale;
    let new_h = h * scale;
    let mut out = vec![0u8; (new_w * new_h * 4) as usize];

    for dst_y in 0..new_h {
        for dst_x in 0..new_w {
            let src_xf = dst_x as f64 * (w as f64 - 1.0) / (new_w as f64 - 1.0).max(1.0);
            let src_yf = dst_y as f64 * (h as f64 - 1.0) / (new_h as f64 - 1.0).max(1.0);

            let x0 = src_xf.floor() as u32;
            let y0 = src_yf.floor() as u32;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);

            let fx = src_xf - x0 as f64;
            let fy = src_yf - y0 as f64;

            for c in 0..4 {
                let p00 = pixels[(y0 * w + x0) as usize * 4 + c] as f64;
                let p10 = pixels[(y0 * w + x1) as usize * 4 + c] as f64;
                let p01 = pixels[(y1 * w + x0) as usize * 4 + c] as f64;
                let p11 = pixels[(y1 * w + x1) as usize * 4 + c] as f64;

                let val = p00 * (1.0 - fx) * (1.0 - fy)
                        + p10 * fx * (1.0 - fy)
                        + p01 * (1.0 - fx) * fy
                        + p11 * fx * fy;

                out[(dst_y * new_w + dst_x) as usize * 4 + c] = val.round() as u8;
            }
        }
    }

    out
}

/// Encode RGBA pixels as a minimal PNG.
fn encode_png(pixels: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut raw_data = Vec::with_capacity((w as usize * 4 + 1) * h as usize);
    for y in 0..h as usize {
        raw_data.push(0); // filter: None
        let row_start = y * w as usize * 4;
        let row_end = row_start + w as usize * 4;
        raw_data.extend_from_slice(&pixels[row_start..row_end]);
    }

    let compressed = deflate_compress(&raw_data);
    // Wrap in zlib: header + compressed + adler32
    let adler = adler32(&raw_data);
    let mut zlib = Vec::new();
    zlib.push(0x78); // CMF
    zlib.push(0x01); // FLG (no dict, level 0)
    zlib.extend_from_slice(&compressed);
    zlib.push((adler >> 24) as u8);
    zlib.push((adler >> 16) as u8);
    zlib.push((adler >> 8) as u8);
    zlib.push(adler as u8);

    let mut png = Vec::new();
    // Signature
    png.extend_from_slice(b"\x89PNG\r\n\x1a\n");

    // IHDR
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.push(8); // bit depth
    ihdr.push(6); // color type RGBA
    ihdr.push(0); // compression
    ihdr.push(0); // filter
    ihdr.push(0); // interlace
    write_png_chunk(&mut png, b"IHDR", &ihdr);

    // IDAT
    write_png_chunk(&mut png, b"IDAT", &zlib);

    // IEND
    write_png_chunk(&mut png, b"IEND", &[]);

    png
}

fn write_png_chunk(png: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    png.extend_from_slice(&(data.len() as u32).to_be_bytes());
    png.extend_from_slice(chunk_type);
    png.extend_from_slice(data);
    let crc = crc32(&[chunk_type.as_slice(), data].concat());
    png.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Minimal deflate compressor using only stored blocks (no actual compression).
/// Keeps WASM binary small; images are still valid PNG.
fn deflate_compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut offset = 0;
    while offset < data.len() {
        let remaining = data.len() - offset;
        let block_size = remaining.min(65535);
        let is_last = offset + block_size >= data.len();

        out.push(if is_last { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00 (stored)
        let len = block_size as u16;
        let nlen = !len;
        out.push(len as u8);
        out.push((len >> 8) as u8);
        out.push(nlen as u8);
        out.push((nlen >> 8) as u8);
        out.extend_from_slice(&data[offset..offset + block_size]);
        offset += block_size;
    }
    if data.is_empty() {
        out.push(0x01); // final empty stored block
        out.extend_from_slice(&[0, 0, 0xFF, 0xFF]);
    }
    out
}

/// Upscale an image using bilinear interpolation.
///
/// # Arguments
/// * `image_bytes` - Raw PNG image bytes
/// * `model` - Model name (unused for bilinear preview, reserved for future)
/// * `scale` - Upscaling factor (2, 3, or 4)
///
/// # Returns
/// PNG-encoded upscaled image bytes
#[wasm_bindgen]
pub fn upscale_preview(image_bytes: &[u8], model: &str, scale: u32) -> Result<Vec<u8>, JsValue> {
    let scale = scale.max(1).min(8);
    let _ = model; // reserved for future model-aware preview

    let (w, h, pixels) = decode_image(image_bytes)
        .map_err(|e| JsValue::from_str(&e))?;

    let upscaled = bilinear_upscale(&pixels, w, h, scale);
    let png = encode_png(&upscaled, w * scale, h * scale);

    Ok(png)
}

/// Returns the version of the WASM preview module.
#[wasm_bindgen]
pub fn version() -> String {
    "0.1.0".to_string()
}
