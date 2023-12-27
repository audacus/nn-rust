use nannou::prelude::*;

use snippets;

use crate::activation::ActivationType;
use crate::data_point::DataPoint;
use crate::gradient_descent::GradientDescent;
use crate::neural_network::NeuralNetwork;

mod activation;
mod data_point;
mod gradient_descent;
mod layer;
mod neural_network;

// Number of generated entries.
const ELEMENTS: usize = 65;
// padding to apply to minimal and maximal values.
const VALUE_PADDING: f32 = 0.025;
// Threshold for changing the color of the points on the grid.
const CLASS_THRESHOLD: f32 = 0.3;
// Layer configuration.
const LAYER_CONFIGURATION: [usize; 3] = [2, 3, 2];
// Activation function.
const ACTIVATION_TYPE: ActivationType = ActivationType::SIGMOID;

// Colors
const COLOR_SAFE: Srgb<u8> = BLUE;
const COLOR_UNSAFE: Srgb<u8> = RED;

// Pixel granularity for drawing boundaries.
const BOUNDARY_STEP: usize = 10;
// Stepping for max chunk size factor.
const MAX_CHUNK_SIZE_FACTOR_STEP: f32 = 0.1;

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
    activation_type: ActivationType,
    gradient_descent: GradientDescent,
    boundary_predictions: Vec<Vec<Option<usize>>>,
    max_chunk_size_factor: f32,
    cost: f32,
    correct: u32,
    learn: bool,
    show_graph: bool,
}

impl GridPoint {
    fn new(x: f32, y: f32, z: f32, color: Srgb<u8>) -> Self {
        GridPoint { x, y, z, color }
    }
}

fn model(app: &App) -> Model {
    app.new_window()
        .key_pressed(key_pressed)
        .build()
        .unwrap();

    let data = get_data_points(ELEMENTS);
    let grid_points = data_to_grid_points(&data);

    let window = app.main_window();
    let win = window.rect();

    let right = win.right() as i32;
    let top = win.top() as i32;

    let center_to_right = (0..right).step_by(BOUNDARY_STEP);
    let center_to_top = (0..top).step_by(BOUNDARY_STEP);

    let boundary_predictions = vec![vec![Some(0); center_to_top.len()]; center_to_right.len()];

    Model {
        points: grid_points,
        data,
        network: NeuralNetwork::new(LAYER_CONFIGURATION.to_vec(), &ACTIVATION_TYPE),
        activation_type: ACTIVATION_TYPE,
        gradient_descent: GradientDescent::new(0.0),
        boundary_predictions,
        max_chunk_size_factor: 0.5,
        cost: 0.0,
        correct: 0,
        learn: false,
        show_graph: false,
    }
}

fn new_run(model: &mut Model) {
    model.learn = false;

    let data = get_data_points(ELEMENTS);
    let grid_points = data_to_grid_points(&data);

    model.data = data;
    model.points = grid_points;
}

fn new_network(model: &mut Model) {
    model.learn = false;

    model.network = NeuralNetwork::new(LAYER_CONFIGURATION.to_vec(), &model.activation_type);
}

fn new_graph(model: &mut Model) {
    let random = snippets::random_numbers().next().unwrap() as f32 / u64::MAX as f32;

    // Get an x value that is max x% to the left / right from the center.
    let x_padding: f32 = 2.5;
    let x: f32 = -x_padding + (random * (2.0 * x_padding));

    model.gradient_descent.past_values = vec![];
    model.gradient_descent.input_value = x;
}

fn update(app: &App, model: &mut Model, _update: Update) {
    if model.learn {
        // Create different chunk sizes from 1 to half.
        for chunk_size in 1..((model.data.len() as f32 * model.max_chunk_size_factor) as usize) {
            for chunk in model.data.chunks(chunk_size) {
                model.network.learn(&chunk.to_vec(), model.gradient_descent.learn_rate, model.gradient_descent.h);
            }
        }
    }

    model.cost = model.network.cost(&model.data);

    model.correct = 0;
    for data_point in &model.data {
        let predicted_class = model.network.classify(data_point.inputs.to_vec());
        if Some(data_point.label) == predicted_class {
            model.correct += 1;
        }
    }

    let window = app.main_window();
    let win = window.rect();

    let right = win.right() as i32;
    let top = win.top() as i32;

    let center_to_right = (0..right).step_by(BOUNDARY_STEP).len();
    let center_to_top = (0..top).step_by(BOUNDARY_STEP).len();

    for x in 0..center_to_right {
        for y in 0..center_to_top {
            let inputs = vec![(x * BOUNDARY_STEP) as f32 / right as f32, (y * BOUNDARY_STEP) as f32 / top as f32];
            model.boundary_predictions[x][y] = model.network.classify(inputs);
        }
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    let window = app.main_window();
    let win = window.rect();

    // Prepare window.
    draw.background().rgb(0.11, 0.12, 0.13);
    draw_crosshair(&draw, &win);

    // Draw data stuff.
    draw_data_points(&draw, &win, model);
    draw_boundries(&draw, &win, model, 2.0);
    draw_info(&draw, &win, model);

    if model.show_graph {
        // Draw graph stuff.
        draw_function_graph(&draw, &win, GradientDescent::function);
        draw_slope(&draw, model, GradientDescent::function);
    }

    draw.to_frame(app, &frame).unwrap();
}

fn draw_info(draw: &Draw, win: &Rect, model: &Model) {
    println!();
    let info_text = format!(
        "cost: {:.10}
learn rate: {:.5}
max chunk size factor: {:.1}
h: {:.10}
correct: {}/{}",
        model.cost, model.gradient_descent.learn_rate, model.max_chunk_size_factor, model.gradient_descent.h, model.correct, model.data.len());

    println!("{}", info_text);

    let pad = 6.0;
    draw.text(&info_text)
        .h(win.pad(pad).h())
        .w(win.pad(pad).w())
        .z(Z_UI)
        .align_text_bottom()
        .left_justify()
        .color(WHITE);
}

fn graph_derivative(x: f32) -> f32 {
    // derivative of: 0.2 * x.powf(4.0) + 0.1 * x.powf(3.0) - x.powf(2.0) + 2.0
    0.8 * x.powf(3.0) + 0.3 * x.powf(2.0) - 2.0 * x
}

fn draw_slope(draw: &Draw, model: &Model, graph_function: fn(f32) -> f32) {
    let past_values = &model.gradient_descent.past_values;
    let x = model.gradient_descent.input_value;
    let y = graph_function(x);

    let slope = graph_derivative(x);

    let weight_x = 2.0;

    // Draw x.
    draw.ellipse()
        .x(x * GRAPH_SCALING)
        .z(Z_UI)
        .color(WHITE)
        .wh(vec2(weight_x, weight_x));

    let weight_point = 8.0;
    // Draw past values.
    for past_x in past_values {
        let point = Vec2::new(past_x * GRAPH_SCALING, graph_function(*past_x) * GRAPH_SCALING);
        draw.ellipse()
            .xy(point)
            .z(Z_GRAPH)
            .color(DARKGRAY)
            .wh(vec2(weight_point, weight_point));
    }
    // Draw point.
    let point = Vec2::new(x * GRAPH_SCALING, y * GRAPH_SCALING);
    draw.ellipse()
        .xy(point)
        .z(Z_UI)
        .color(WHITE)
        .wh(vec2(weight_point, weight_point));

    // Draw point text.
    let slope_text = format!("{:.2}", slope);
    draw.text(&slope_text)
        .xy(point + vec2(0.0, 20.0))
        .z(Z_UI)
        .color(WHITE);

    // Visualize the slope.
    let slope_direction = Vec2::new(1.0, slope).normalize();

    // Draw slope.
    let weight_slope = 2.0;
    draw.line()
        .start(point - (slope_direction * GRAPH_SCALING))
        .end(point + (slope_direction * GRAPH_SCALING))
        .z(Z_GRAPH)
        .color(RED)
        .weight(weight_slope);
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

fn draw_data_points(draw: &Draw, win: &Rect, model: &Model) {
    // Draw grid points.
    for point in &model.points {
        draw.ellipse()
            .wh(vec2(10.0, 10.0))
            .x(point.x * win.right())
            .y(point.y * win.top())
            .z(point.z)
            .color(point.color);
    }
}

fn draw_crosshair(draw: &Draw, win: &Rect) {
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
}

fn draw_boundries(draw: &Draw, win: &Rect, model: &Model, weight: f32) {
    let right = win.right() as i32;
    let top = win.top() as i32;

    let center_to_right = (0..right).step_by(BOUNDARY_STEP).len();
    let center_to_top = (0..top).step_by(BOUNDARY_STEP).len();

    for x in 0..center_to_right {
        for y in 0..center_to_top {
            let predicted_class = model.boundary_predictions[x][y];

            let pixel = draw.ellipse()
                .xyz(vec3((x * BOUNDARY_STEP) as f32, (y * BOUNDARY_STEP) as f32, Z_BOUNDRY))
                .wh(vec2(weight, weight));

            if predicted_class == Some(0) {
                pixel.color(COLOR_SAFE);
            } else if predicted_class == Some(1) {
                pixel.color(COLOR_UNSAFE);
            }
        }
    }
}

fn data_to_grid_points(data: &Vec<DataPoint>) -> Vec<GridPoint> {
    let mut grid_points: Vec<GridPoint> = Vec::with_capacity(data.len());

    for data_point in data {
        grid_points.push(GridPoint::new(
            data_point.inputs[0],
            data_point.inputs[1],
            Z_POINTS,
            if data_point.label == 1 { COLOR_UNSAFE } else { COLOR_SAFE },
        ));
    }

    grid_points
}

fn get_data_points(elements: usize) -> Vec<DataPoint> {
    let mut data: Vec<DataPoint> = Vec::new();

    for _ in 0..elements {
        // Get random float values between 0 and 1 for x and y.
        let x = snippets::random_numbers().next().unwrap() as f32 / u64::MAX as f32;
        let y = snippets::random_numbers().next().unwrap() as f32 / u64::MAX as f32;

        // Apply padding to the axis values.
        let min = 0.0 + VALUE_PADDING;
        let max = 1.0 - VALUE_PADDING;
        let spot_size = min + (x * (max - min));
        let spike_length = min + (y * (max - min));

        // Set poisenous flag based on the class threshold.
        let poisenous = spot_size.pow(0.8) * spike_length.pow(0.8) > CLASS_THRESHOLD;

        data.push(DataPoint::new(
            vec![spot_size, spike_length],
            if poisenous { 1 } else { 0 },
            2,
        ));
    }

    data
}

fn key_pressed(_app: &App, model: &mut Model, key: Key) {
    // println!("key pressed: {:?}", key);

    match key {
        // New
        Key::N => new_run(model),
        // New network.
        Key::S => new_network(model),
        // Network learn iteration.
        Key::I => model.network.learn(&model.data, model.gradient_descent.learn_rate, model.gradient_descent.h),
        // Toggle learning.
        Key::Space => model.learn = !model.learn,
        // Learn rate.
        Key::Up => model.gradient_descent.learn_rate += GradientDescent::LEARN_RATE_STEP,
        Key::Down => model.gradient_descent.learn_rate -= GradientDescent::LEARN_RATE_STEP,
        // Max chunk size factor.
        Key::K => model.max_chunk_size_factor += MAX_CHUNK_SIZE_FACTOR_STEP,
        Key::J => model.max_chunk_size_factor -= MAX_CHUNK_SIZE_FACTOR_STEP,
        // H.
        Key::Right => model.gradient_descent.h /= GradientDescent::H_FACTOR,
        Key::Left => model.gradient_descent.h *= GradientDescent::H_FACTOR,
        // Toggle graph.
        Key::G => model.show_graph = !model.show_graph,
        // New graph.
        Key::R => new_graph(model),
        // Graph learn iteration.
        Key::L => model.gradient_descent.learn(),
        _ => (),
    }

    model.gradient_descent.learn_rate = model.gradient_descent.learn_rate.clamp(0.0, 100.0);
    model.max_chunk_size_factor = model.max_chunk_size_factor.clamp(0.0, 1.0);
    model.gradient_descent.h = model.gradient_descent.h.clamp(0.0, 1.0);

    println!("learn rate: {:.5}", model.gradient_descent.learn_rate);
    println!("h: {:.10}", model.gradient_descent.h);
}