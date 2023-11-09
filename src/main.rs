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

    // Debug entries.
    // for entry in &entries {
    //     let predicted_class = classify(entry.spike_length, entry.spot_size);
    //     println!("spot size: {} spike length: {} poisenous: {} -> {}", entry.spot_size, entry.spike_length, entry.poisenous, predicted_class);
    // }

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

    /*
    // 100-step and 10-step grids.
    draw_grid(&draw, &win, 100.0, 1.0);
    draw_grid(&draw, &win, 25.0, 0.5);
     */

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

    /*
    // Crosshair text.
    let font_size = 14;
    let top = format!("{:.1}", win.top());
    let bottom = format!("{:.1}", win.bottom());
    let left = format!("{:.1}", win.left());
    let right = format!("{:.1}", win.right());
    let x_off = 30.0;
    let y_off = 20.0;
    draw.text("0.0")
        .x_y(15.0, 15.0)
        .color(crosshair_color)
        .font_size(font_size);
    draw.text(&top)
        .h(win.h())
        .font_size(font_size)
        .align_text_top()
        .color(crosshair_color)
        .x(x_off);
    draw.text(&bottom)
        .h(win.h())
        .font_size(font_size)
        .align_text_bottom()
        .color(crosshair_color)
        .x(x_off);
    draw.text(&left)
        .w(win.w())
        .font_size(font_size)
        .left_justify()
        .color(crosshair_color)
        .y(y_off);
    draw.text(&right)
        .w(win.w())
        .font_size(font_size)
        .right_justify()
        .color(crosshair_color)
        .y(y_off);
    */

    /*
    let mouse_position = app.mouse.position();

    // Ellipse at mouse.
    draw.ellipse()
        .wh([5.0; 2].into())
        .xy(mouse_position)
        .z(1.0);

    // Mouse position text.
    let position_text = format!("[{:.1}, {:.1}]", mouse_position.x, mouse_position.y);
    draw.text(&position_text)
        .xy(mouse_position + vec2(0.0, 20.0))
        .z(1.0)
        .font_size(font_size)
        .color(WHITE);
     */

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

fn draw_grid(draw: &Draw, win: &Rect, step: f32, weight: f32) {
    // x axis
    let step_by = || (0..).map(|i| i as f32 * step);
    let r_iter = step_by().take_while(|&f| f < win.right());
    let l_iter = step_by().map(|f| -f).take_while(|&f| f > win.left());
    let x_iter = r_iter.chain(l_iter);

    for x in x_iter {
        draw.line()
            .weight(weight)
            .points(pt2(x, win.bottom()), pt2(x, win.top()));
    }

    // y axis
    let t_iter = step_by().take_while(|&f| f < win.top());
    let b_iter = step_by().map(|f| -f).take_while(|&f| f > win.bottom());
    let y_iter = t_iter.chain(b_iter);

    for y in y_iter {
        draw.line()
            .weight(weight)
            .points(pt2(win.left(), y), pt2(win.right(), y));
    }
}