use snippets;
use nannou::prelude::*;

// Weights for connections to the first output.
const WEIGHT_1_1: f32 = 0.0;
const WEIGHT_1_2: f32 = 0.0;
// Weights for connections to the second output.
const WEIGHT_2_1: f32 = 0.0;
const WEIGHT_2_2: f32 = 0.0;
// Bias values.
const BIAS_1: f32 = 0.0;
const BIAS_2: f32 = 0.0;

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

fn main() {
    nannou::app(model)
        .update(update)
        .simple_window(view)
        .run();
}

struct Fruit {
    spike_length: f32,
    spot_size: f32,
    poisenous: bool,
}

fn classify(input_1: f32, input_2: f32) -> u8 {
    let output_1 = input_1 * WEIGHT_1_1 + input_2 * WEIGHT_2_1 + BIAS_1;
    let output_2 = input_1 * WEIGHT_1_2 + input_2 * WEIGHT_2_2 + BIAS_2;

    if output_1 > output_2 { 0 } else { 1 }
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
}

impl GridPoint {
    fn new(x: f32, y: f32, z: f32, color: Srgb<u8>) -> Self {
        GridPoint { x, y, z, color }
    }
}

fn model (_app: &App) -> Model {
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

    Model {
        points: grid_points
    }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {}

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


    draw_boundries(&draw, &win, 10, 2.0);

    draw.to_frame(app, &frame).unwrap();
}

fn draw_boundries(draw: &Draw, win: &Rect, step: usize, weight: f32) {
    let left = win.left() as i32;
    let right = win.right() as i32;
    let bottom = win.bottom() as i32;
    let top = win.top() as i32;

    for x in (left..right).step_by(step) {
        for y in (bottom..top).step_by(step) {
            let predicted_class = classify(x as f32 / (left - right) as f32, y as f32 / (bottom - top) as f32);

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