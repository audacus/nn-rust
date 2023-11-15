use snippets;
use nannou::prelude::*;

use crate::data_point::DataPoint;
use crate::neural_network::NeuralNetwork;

mod data_point;
mod layer;
mod neural_network;

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
const Z_POINTS: f32 = 2.0;
const Z_GRAPH: f32 = 3.0;
const Z_UI: f32 = 4.0;

// 100 pixel correspond to value 1.0
const GRAPH_SCALING: f32 = 100.0;

fn main() {

    nannou::app(model)
        .update(update)
        .view(view)
        .run();
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
    data: Vec<DataPoint>,
    network: NeuralNetwork,
}

impl GridPoint {
    fn new(x: f32, y: f32, z: f32, color: Srgb<u8>) -> Self {
        GridPoint { x, y, z, color }
    }
}

fn model (app: &App) -> Model {
    // Create a list with n random entries.
    let mut data: Vec<DataPoint> = Vec::new();
    for _ in 0..ELEMENTS {
        let mut spot_size = snippets::random_numbers().next().unwrap() as f32 / std::u64::MAX as f32;
        let mut spike_length = snippets::random_numbers().next().unwrap() as f32 / std::u64::MAX as f32;

        let min = 0.0 + VALUE_PADDING;
        let max = 1.0 - VALUE_PADDING;
        spot_size = min + (spot_size * (max - min));
        spike_length = min + (spike_length * (max - min));

        let poisenous = (spot_size + spike_length) > POISENOUS_THRESHOLD;

        data.push(DataPoint::new(
            vec![spot_size, spike_length],
            if poisenous { 1 } else { 0 },
            2,
        ));
    }

    // Transform entries to grid points.
    let mut grid_points: Vec<GridPoint> = Vec::new();
    for entry in &data {
        grid_points.push(GridPoint::new(
            entry.inputs[0],
            entry.inputs[1],
            Z_POINTS,
            if entry.label == 1 { COLOR_POISENOUS } else { COLOR_SAFE },
        ));
    }

    app.new_window()
        .key_pressed(key_pressed)
        .build()
        .unwrap();

    Model {
        points: grid_points,
        data,
        network: NeuralNetwork::new(vec![2, 3, 2]),
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
            .wh(vec2(10.0, 10.0))
            .x(point.x * win.right())
            .y(point.y * win.top())
            .z(point.z)
            .color(point.color);
    }

    draw_boundries(&draw, &win, model, 15, 2.0);

    show_cost(&draw, &win, model);

    // Draw graph stuff.
    draw_function_graph(&draw, &win, graph_function);
    draw_slope(&app, &draw, graph_function);

    draw.to_frame(app, &frame).unwrap();
}

fn show_cost(draw: &Draw, win: &Rect, model: &Model) {
    let cost = model.network.cost(&model.data);
    let cost_text = format!("cost: {:.4}", cost);

    let pad = 6.0;
    draw.text(&cost_text)
        .h(win.pad(pad).h())
        .w(win.pad(pad).w())
        .z(Z_UI)
        .align_text_bottom()
        .left_justify()
        .color(WHITE);
}

fn graph_function(x: f32) -> f32 {
    0.2 * pow(x, 4) + 0.1 * pow(x, 3) - pow(x, 2) + 2.0
}

fn  draw_slope(app: &App, draw: &Draw, graph_function: fn(f32) -> f32) {
    // Get x from mouse position.
    let mouse = app.mouse.position();
    let x = mouse.x / GRAPH_SCALING;
    let y = graph_function(x);

    // Aproximate the slope of the function at x.
    let h = 0.00001;
    let delta_output = graph_function(x + h) - y;
    let slope = delta_output / h;

    // Visualize the slope.
    let slope_direction = Vec2::new(1.0, slope).normalize();
    let point = Vec2::new(x, y);

    // Draw slope.
    let weight_slope = 2.0;
    draw.line()
        .start((point - slope_direction) * GRAPH_SCALING)
        .end((point + slope_direction) * GRAPH_SCALING)
        .z(Z_GRAPH)
        .color(RED)
        .weight(weight_slope);

    // Draw x.
    let weight = 8.0;
    draw.ellipse()
        .xy(point * GRAPH_SCALING)
        .z(Z_UI)
        .color(WHITE)
        .wh(vec2(weight, weight));

    // Draw x text.
    let slope_text = format!("{:.2}", slope);
    draw.text(&slope_text)
        .xy(point * GRAPH_SCALING + vec2(0.0, 20.0))
        .z(Z_UI)
        .color(WHITE);
}

fn draw_function_graph(draw: &Draw, win: &Rect, graph_function: fn(f32) -> f32) {
    let left = win.left() as i32;
    let right = win.right() as i32;

    // Create vector with points for every pixel from left to right (x axis).
    let mut points: Vec<Point3> = Vec::with_capacity((left..right).count());
    for x in left..right {
        let y = graph_function(x as f32 / GRAPH_SCALING) * GRAPH_SCALING;
        points.push(pt3(x as f32, y, Z_GRAPH));
    }

    // Draw line with points.
    draw.polyline()
        .weight(1.0)
        .points(points)
        .color(STEELBLUE);
}

fn draw_boundries(draw: &Draw, win: &Rect, model: &Model, step: usize, weight: f32) {

    let left = win.left() as i32;
    let right = win.right() as i32;
    let bottom = win.bottom() as i32;
    let top = win.top() as i32;

    for x in (left..right).step_by(step) {
        for y in (bottom..top).step_by(step) {

            let inputs = vec![x as f32 / right as f32, y as f32 / top as f32];
            let predicted_class = model.network.classify(&inputs);

            let pixel = draw.ellipse()
                .xyz(vec3(x as f32, y as f32, Z_BOUNDRY))
                .wh(vec2(weight, weight));


            if predicted_class == Some(0) {
                pixel.color(COLOR_SAFE);
            } else if predicted_class == Some(1) {
                pixel.color(COLOR_POISENOUS);
            }
        }
    }
}

fn key_pressed(_app: &App, _model: &mut Model, key: Key) {
    println!("key pressed: {:?}", key);
}
