use rand;
use rand::Rng;
use nannou::prelude::*;
use nannou_egui::Egui;
use nannou_egui::{self, egui};
use std::sync::{Arc, Mutex};
use std::thread;

pub struct Model {
    pub sample_n: usize,
    pub samples: Vec<Point2>,
    pub hg_g: f32,
    pub alpha: f32,
    pub length: f32,
    pub dir: Point2,
    pub egui: Egui,
}

fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn transform(wi: &Point2, dir: Point2) -> Point2 {
    let dir_x = wi.y * dir.x + wi.x * dir.y;
    let dir_y = -wi.x * dir.x + wi.y * dir.y;
    pt2(dir_x, dir_y)
}

fn get_sample(wi: &Point2, g: f32) -> Point2 {
    let next_rd:f32 = rand::thread_rng().gen_range(-1.0..1.0);
    let tan_val = (std::f32::consts::FRAC_PI_2 * next_rd).tan();
    let inner = (1. - g) * tan_val / (1.000001 + g);
    if inner.is_infinite() || inner.is_nan() {
        println!("NaN, next rd = {}", next_rd);
    }
    let sign: f32 = match rand::random::<bool>() {true => 1., false => -1.};
    let cos_t = (2. * inner.atan()).cos().min(1.);
    let sin_t = sign * (1. - cos_t * cos_t).max(0.).sqrt();
    let pt = pt2(sin_t, cos_t);
    transform(wi, pt)
}

pub fn model(app: &App) -> Model {
    let win_id = app
        .new_window()
        .event(event)
        .raw_event(raw_window_event)
        .size(800, 800)
        .view(view)
        .build().unwrap();
    Model {
        sample_n: 11, samples: vec![pt2(0., 0.); 2048], dir: pt2(0., 0.), hg_g: 0.5, 
        alpha: 0.5, length: 200., egui: Egui::from_window(&app.window(win_id).unwrap()),
    }
}

pub fn update(_app: &App, model: &mut Model, _update: Update) {
    update_gui(_app, model, &_update);
    model.dir = -_app.mouse.position().normalize_or_zero();
    let pts = Arc::new(Mutex::new(Vec::<Point2>::new()));
    let wi = Arc::new(model.dir);
    let mut threads = vec![];
    let sample_n = model.sample_n;
    let hg_coeff = model.hg_g;
    for _ in 0..8 {
        threads.push(thread::spawn({
            let clone = Arc::clone(&pts);
            let local_wi = Arc::clone(&wi);
            move || {
                let pnum = 1 << (sample_n - 3);
                let thread_wi: Vec2 = *local_wi;
                for _ in 0..pnum {
                    let sample = get_sample(&thread_wi, hg_coeff);
                    let mut v = clone.lock().unwrap();
                    v.push(sample);
                }
            }
        }));
    }
    for t in threads {
        t.join().unwrap();
    }
    let inner = pts.lock().unwrap();
    model.samples.clear();
    for pt in inner.iter() {
        model.samples.push(*pt);
    }
}

fn event(_app: &App, _model: &mut Model, _event: WindowEvent) {}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();
    draw.background().rgba(0., 0., 0., 1.0);
    draw.arrow()
        .start(-model.dir * model.length)
        .end(pt2(0., 0.))
        .weight(4.)
        .color(MEDIUMSPRINGGREEN);
    
    for pt in model.samples.iter() {
        draw.arrow()
            .start(pt2(0., 0.))
            .end(*pt * model.length)
            .weight(1.)
            .rgba(1., 0., 0., model.alpha);
    }

    draw.to_frame(app, &frame).unwrap();
    model.egui.draw_to_frame(&frame).unwrap();
}

pub fn update_gui(_app: &App, model: &mut Model, update: &Update) {
    let Model {
        ref mut sample_n,
        ref mut length,
        ref mut hg_g,
        ref mut alpha,
        ref mut egui,
        ..
    } = model;
    egui.set_elapsed_time(update.since_start);
    let ctx = egui.begin_frame();
    let window = egui::Window::new("Configuration").default_width(200.);
    window.show(&ctx, |ui| {
        egui::Grid::new("slide_bars")
            .striped(true)
        .show(ui, |ui| {
            ui.label("sample n (power 2): ");
            ui.add(egui::Slider::new(sample_n, 8..=14));
            ui.end_row();
            ui.label("HG coefficient: ");
            ui.add(egui::Slider::new(hg_g, -1.0..=1.0));
            ui.end_row();
            ui.label("Length: ");
            ui.add(egui::Slider::new(length, 150.0..=350.));
            ui.end_row();
            ui.label("Alpha: ");
            ui.add(egui::Slider::new(alpha, 0.001..=0.9));
            ui.end_row();
        });
    });
}
