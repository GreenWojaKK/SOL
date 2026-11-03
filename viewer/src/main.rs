//! `vppcontract probabilistic` 결과 폴더를 보는 확률예측 비교 viewer.
//!
//! ```text
//! probabilistic-viewer [결과 폴더 또는 explorer-data.json]
//! ```
//!
//! 경로를 주지 않으면 빈 창에서 결과 폴더를 고른다.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod charts;
mod data;
mod explorer;
mod number;

use std::path::PathBuf;

use gpui::{
    App, Bounds, KeyBinding, TitlebarOptions, WindowBounds, WindowOptions, prelude::*, px, size,
};
use gpui_platform::application;

use crate::explorer::{Explorer, OpenFolder};

fn main() {
    let path = std::env::args_os().nth(1).map(PathBuf::from);
    application().run(move |cx: &mut App| {
        cx.bind_keys([KeyBinding::new("ctrl-o", OpenFolder, None)]);
        cx.on_window_closed(|cx, _| {
            if cx.windows().is_empty() {
                cx.quit();
            }
        })
        .detach();

        let bounds = Bounds::centered(None, size(px(1440.), px(960.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                titlebar: Some(TitlebarOptions {
                    title: Some("확률예측 비교".into()),
                    ..Default::default()
                }),
                ..Default::default()
            },
            move |window, cx| cx.new(|cx| Explorer::new(path, window, cx)),
        )
        .expect("창을 열 수 없습니다");
        cx.activate(true);
    });
}
