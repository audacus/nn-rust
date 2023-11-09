use snippets;
use nannou::prelude::*;

// Number of generated entries.
const ELEMENTS: u8 = 100;
// padding to apply to minimal and maximal values.
const VALUE_PADDING: f32 = 0.025;
// Threshold for changing the color of the points on the grid.
const POISENOUS_THRESHOLD: f32 = 0.66;

// Colors
const COLOR_SAFE: Srgb<u8> = BLUE;
const COLOR_POISENOUS: Srgb<u8> = RED;

// z indexes
const Z_BOUNDRY: f32 = 1.0;
const Z_POINTS: f32 = 10.0;

// Offsets
const OFFSET_WEIGHT: f32 = 0.01;
const OFFSET_BIAS: f32 = 0.001;

fn main() {
    nannou::app(model)
        .update(update)
        .view(view)
        .run();
}

struct Fruit {
    spike_length: f32,
    spot_size: f32,
    poisenous: bool,
}

// Draw stuff.
struct GridPoint {
    x: f32,
    y: f32,
    z: f32,
    color: Srgb<u8>,
}
struct Model {
    points: Vec<GridPoint>,
    // Weights for connections to the first output.
    weight_1_1: f32,
    weight_1_2: f32,
    // Weights for connections to the second output.
    weight_2_1: f32,
    weight_2_2: f32,
    // Bias values.
    bias_1: f32,
    bias_2: f32,
}

impl GridPoint {
    fn new(x: f32, y: f32, z: f32, color: Srgb<u8>) -> Self {
        GridPoint { x, y, z, color }
    }
}

fn model (app: &App) -> Model {
    // Create a list with n random entries.
    let mut entries: Vec<Fruit> = Vec::new();
    for _ in 0..ELEMENTS {
        let mut spot_size = snippets::random_numbers().next().unwrap() as f32 / std::u64::MAX as f32;
        let mut spike_length = snippets::random_numbers().next().unwrap() as f32 / std::u64::MAX as f32;

        let min = 0.0 + VALUE_PADDING;
        let max = 1.0 - VALUE_PADDING;
        spot_size = min + (spot_size * (max - min));
        spike_length = min + (spike_length * (max - min));

        entries.push(Fruit {
            spot_size,
            spike_length,
            poisenous: (spot_size + spike_length) > POISENOUS_THRESHOLD,
        });
    }

    // Transform entries to grid points.
    let mut grid_points: Vec<GridPoint> = Vec::new();
    for entry in &entries {
        grid_points.push(GridPoint::new(
            entry.spot_size,
            entry.spike_length,
            Z_POINTS,
            if entry.poisenous { COLOR_POISENOUS } else { COLOR_SAFE },
        ));
    }

    app.new_window()
        .key_pressed(key_pressed)
        .build()
        .unwrap();

    Model {
        points: grid_points,
        weight_1_1: 0.0,
        weight_1_2: 0.0,
        weight_2_1: 0.0,
        weight_2_2: 0.0,
        bias_1: 0.0,
        bias_2: 0.0,
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    let window = app.main_window();
    let win = window.rect();

    draw.background().rgb(0.11, 0.12, 0.13);

    // Crosshair.
    let crosshair_color = gray(0.5);
    let ends = [
        win.mid_top(),
        win.mid_right(),
        win.mid_bottom(),
        win.mid_left(),
    ];

    for &end in &ends {
        draw.line()
            .color(crosshair_color)
            .end(end);
    }

    // Draw grid points.
    for point in &model.points {
        draw.ellipse()
            .wh([10.0, 10.0].into())
            .x(point.x * win.right())
            .y(point.y * win.top())
            .z(point.z)
            .color(point.color);
    }


    draw_boundries(&draw, &win, model, 10, 2.0);

    draw.to_frame(app, &frame).unwrap();
}

fn draw_boundries(draw: &Draw, win: &Rect, model: &Model, step: usize, weight: f32) {
    let left = win.left() as i32;
    let right = win.right() as i32;
    let bottom = win.bottom() as i32;
    let top = win.top() as i32;

    for x in (left..right).step_by(step) {
        for y in (bottom..top).step_by(step) {
            let predicted_class = classify(model, x as f32 / (left - right) as f32, y as f32 / (bottom - top) as f32);

            let pixel = draw.rect()
            .xyz(vec3(x as f32, y as f32, Z_BOUNDRY))
            .wh(vec2(weight, weight));

            if predicted_class == 0 {
                pixel.color(COLOR_SAFE);
            } else if predicted_class == 1 {
                pixel.color(COLOR_POISENOUS);
            }
        }
    }
}

fn classify(model: &Model, input_1: f32, input_2: f32) -> u8 {
    let output_1 = input_1 * model.weight_1_1 + input_2 * model.weight_1_2 + model.bias_1;
    let output_2 = input_1 * model.weight_2_1 + input_2 * model.weight_2_2 + model.bias_2;

    if output_1 > output_2 { 0 } else { 1 }
}

fn key_pressed(_app: &App, model: &mut Model, key: Key) {
    match key {
        // Weight 1 1
        Key::J => model.weight_1_1 -= OFFSET_WEIGHT,
        Key::K => model.weight_1_1 += OFFSET_WEIGHT,
        // Weight 1 2
        Key::H => model.weight_1_2 -= OFFSET_WEIGHT,
        Key::L => model.weight_1_2 += OFFSET_WEIGHT,
        // Weight 2 1
        Key::S => model.weight_2_1 -= OFFSET_WEIGHT,
        Key::D => model.weight_2_1 += OFFSET_WEIGHT,
        // Weight 2 2
        Key::A => model.weight_2_2 -= OFFSET_WEIGHT,
        Key::F => model.weight_2_2 += OFFSET_WEIGHT,
        // Bias 1
        Key::Up => model.bias_1 += OFFSET_BIAS,
        Key::Down => model.bias_1 -= OFFSET_BIAS,
        // Bias 2
        Key::Left => model.bias_2 -= OFFSET_BIAS,
        Key::Right => model.bias_2 += OFFSET_BIAS,
        _ => {}
    };

    model.weight_1_1 = model.weight_1_1.clamp(-1.0, 1.0);
    model.weight_1_1 = model.weight_1_1.clamp(-1.0, 1.0);
    model.weight_1_2 = model.weight_1_2.clamp(-1.0, 1.0);
    model.weight_1_2 = model.weight_1_2.clamp(-1.0, 1.0);
    model.weight_2_1 = model.weight_2_1.clamp(-1.0, 1.0);
    model.weight_2_1 = model.weight_2_1.clamp(-1.0, 1.0);
    model.weight_2_2 = model.weight_2_2.clamp(-1.0, 1.0);
    model.weight_2_2 = model.weight_2_2.clamp(-1.0, 1.0);
    model.bias_1 = model.bias_1.clamp(-1.0, 1.0);
    model.bias_1 = model.bias_1.clamp(-1.0, 1.0);
    model.bias_2 = model.bias_2.clamp(-1.0, 1.0);
    model.bias_2 = model.bias_2.clamp(-1.0, 1.0);

    println!("weight 1 1: {:.2}", model.weight_1_1);
    println!("weight 1 2: {:.2}", model.weight_1_2);
    println!("weight 2 1: {:.2}", model.weight_2_1);
    println!("weight 2 2: {:.2}", model.weight_2_2);
    println!("bias 1: {:.3}", model.bias_1);
    println!("bias 2: {:.3}", model.bias_2);
    println!("---");

}
