//! 확률예측 비교 화면. 결과 폴더 하나를 열어 날짜별 예측구간, 계절별 표, 계절·시간대 차트,
//! 예보 조건별 표를 보여준다.

use std::path::PathBuf;

use gpui::{
    AnyElement, ClickEvent, Context, Div, Entity, FocusHandle, PathPromptOptions, Pixels, Point,
    SharedString, Stateful, Subscription, WeakEntity, Window, actions, div, point, prelude::*, px,
    relative,
};
use viewer_ui::menu::{self, Menu, MenuEvent};
use viewer_ui::table::{self, Cell, ColumnSpec};
use viewer_ui::theme;
use viewer_ui::widgets::{small_button, toggle_button};

use crate::charts::{self, SeasonMetric};
use crate::data::{self, GROUPS, MEDIAN, Report, season_name};
use crate::number::{int, num, pct};

actions!(explorer, [OpenFolder]);

enum Status {
    Empty,
    Loading(PathBuf),
    Failed(SharedString),
    Ready,
}

/// 마우스가 올라간 차트 위치.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hover {
    /// 날짜 차트 순번과 시각 순번.
    Fan { panel: usize, hour: usize },
    /// 히트맵 순번과 (계절, 시각) 칸.
    Heat { panel: usize, cell: (usize, usize) },
}

impl Hover {
    fn panel(&self) -> (bool, usize) {
        match self {
            Self::Fan { panel, .. } => (true, *panel),
            Self::Heat { panel, .. } => (false, *panel),
        }
    }
}

/// 접고 펴는 구역.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Section {
    Reliability,
    Hourly,
    Regimes,
}

pub struct Explorer {
    focus_handle: FocusHandle,
    report: Option<Report>,
    status: Status,
    /// 보고 있는 날짜 순번.
    date: usize,
    /// 날짜 차트와 표에 보일 그룹. `None`은 두 그룹 모두다.
    group: Option<&'static str>,
    /// 계절별 표의 계절. `all`은 전체 기간이다.
    season: String,
    /// 열린 날짜 고르기 메뉴와 그 위치.
    date_menu: Option<(Entity<Menu>, Point<Pixels>, Subscription)>,
    hover: Option<Hover>,
    open: Vec<Section>,
    title: Option<String>,
    /// 운영체제의 앱 색상 설정을 따르는지. 화면 전환 버튼을 누르면 그 뒤로는 따르지 않는다.
    follow_system: bool,
    _subscriptions: Vec<Subscription>,
}

const SUMMARY: [ColumnSpec; 8] = [
    ("모델 · 그룹", 2.4),
    ("유효 일수", 1.0),
    ("주간 시각 수", 1.1),
    ("잔차 중앙값 kWh", 1.4),
    ("잔차 평균 kWh", 1.4),
    ("Pinball kWh", 1.3),
    ("W 백만원", 1.2),
    ("원출력 crossing", 1.3),
];

const INTERVALS: [ColumnSpec; 7] = [
    ("모델 · 그룹", 2.4),
    ("명목 coverage", 1.2),
    ("실제 coverage", 1.2),
    ("평균 폭 / 용량", 1.3),
    ("하단 미만", 1.1),
    ("상단 초과", 1.1),
    ("경계값 일치", 1.1),
];

const REGIMES: [ColumnSpec; 7] = [
    ("조건", 1.4),
    ("모델 · 그룹", 2.4),
    ("시각 수", 1.0),
    ("잔차 중앙값 kWh", 1.4),
    ("80% coverage", 1.2),
    ("80% 폭 / 용량", 1.2),
    ("Pinball kWh", 1.2),
];

impl Explorer {
    pub fn new(path: Option<PathBuf>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);
        theme::set_mode(theme::Mode::from_appearance(window.appearance()));
        let appearance = cx.observe_window_appearance(window, |this, window, cx| {
            if this.follow_system {
                theme::set_mode(theme::Mode::from_appearance(window.appearance()));
                cx.notify();
            }
        });
        let mut explorer = Self {
            focus_handle,
            report: None,
            status: Status::Empty,
            date: 0,
            group: None,
            season: "all".into(),
            date_menu: None,
            hover: None,
            open: Vec::new(),
            title: None,
            follow_system: true,
            _subscriptions: vec![appearance],
        };
        if let Some(path) = path {
            explorer.load(path, cx);
        }
        explorer
    }

    /// 결과 폴더를 백그라운드에서 읽는다.
    fn load(&mut self, path: PathBuf, cx: &mut Context<Self>) {
        self.status = Status::Loading(path.clone());
        cx.notify();
        cx.spawn(async move |this, cx| {
            let result = cx.background_spawn(async move { data::load(&path) }).await;
            this.update(cx, |this, cx| {
                match result {
                    Ok(report) => {
                        this.report = Some(report);
                        this.status = Status::Ready;
                        this.date = 0;
                        this.season = "all".into();
                        this.hover = None;
                    }
                    Err(message) => this.status = Status::Failed(message.into()),
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    fn open_folder(&mut self, _: &OpenFolder, _window: &mut Window, cx: &mut Context<Self>) {
        let selection = cx.prompt_for_paths(PathPromptOptions {
            files: false,
            directories: true,
            multiple: false,
            prompt: Some("결과 폴더 열기".into()),
        });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = selection.await
                && let Some(path) = paths.into_iter().next()
            {
                this.update(cx, |this, cx| this.load(path, cx)).ok();
            }
        })
        .detach();
    }

    fn move_date(&mut self, delta: isize, cx: &mut Context<Self>) {
        let Some(report) = &self.report else {
            return;
        };
        let last = report.dates.len().saturating_sub(1) as isize;
        self.date = (self.date as isize + delta).clamp(0, last) as usize;
        self.hover = None;
        cx.notify();
    }

    fn set_date(&mut self, date: usize, cx: &mut Context<Self>) {
        self.date = date;
        self.hover = None;
        cx.notify();
    }

    /// 날짜 고르기의 첫 단계: 보고서에 있는 달 목록. 달을 고르면 같은 자리에 그 달의 날짜 목록이 뜬다.
    fn open_month_menu(
        &mut self,
        position: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(report) = &self.report else {
            return;
        };
        let mut months: Vec<(String, usize)> = Vec::new();
        for date in &report.dates {
            let month = date.get(..7).unwrap_or(date).to_string();
            match months.last_mut() {
                Some((last, count)) if *last == month => *count += 1,
                _ => months.push((month, 1)),
            }
        }
        let current = report
            .dates
            .get(self.date)
            .and_then(|date| date.get(..7))
            .map(str::to_string);
        let this = cx.entity().downgrade();
        let menu = cx.new(|cx| {
            let mut menu = Menu::new(cx);
            for (month, count) in months {
                let mark = if current.as_deref() == Some(month.as_str()) {
                    "✓ "
                } else {
                    "   "
                };
                let label = format!("{mark}{month} · {count}일");
                let this = this.clone();
                menu = menu.item(label, move |window, cx| {
                    let month = month.clone();
                    this.update(cx, |this, cx| {
                        this.open_day_menu(&month, position, window, cx)
                    })
                    .ok();
                });
            }
            menu
        });
        self.show_date_menu(menu, position, window, cx);
    }

    /// 날짜 고르기의 둘째 단계: 고른 달의 날짜 목록.
    fn open_day_menu(
        &mut self,
        month: &str,
        position: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(report) = &self.report else {
            return;
        };
        let days: Vec<(usize, String)> = report
            .dates
            .iter()
            .enumerate()
            .filter(|(_, date)| date.starts_with(month))
            .map(|(ix, date)| (ix, date.clone()))
            .collect();
        let current = self.date;
        let this = cx.entity().downgrade();
        let menu = cx.new(|cx| {
            let mut menu = Menu::new(cx);
            for (ix, date) in days {
                let mark = if ix == current { "✓ " } else { "   " };
                let this = this.clone();
                menu = menu.item(format!("{mark}{date}"), move |_, cx| {
                    this.update(cx, |this, cx| this.set_date(ix, cx)).ok();
                });
            }
            menu
        });
        self.show_date_menu(menu, position, window, cx);
    }

    fn show_date_menu(
        &mut self,
        menu: Entity<Menu>,
        position: Point<Pixels>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        menu.update(cx, |menu, cx| menu.focus(window, cx));
        let subscription = cx.subscribe_in(
            &menu,
            window,
            |this, closed, MenuEvent::Dismissed, _, cx| {
                // 달을 고르면 날짜 목록이 먼저 열린 뒤 달 목록이 닫힌다. 지금 열린 메뉴가 닫힐 때만 지운다.
                if this
                    .date_menu
                    .as_ref()
                    .is_some_and(|(open, _, _)| open == closed)
                {
                    this.date_menu = None;
                    cx.notify();
                }
            },
        );
        self.date_menu = Some((menu, position, subscription));
        cx.notify();
    }

    fn set_hover(
        &mut self,
        kind_fan: bool,
        panel: usize,
        hover: Option<Hover>,
        cx: &mut Context<Self>,
    ) {
        let next = match hover {
            Some(hover) => Some(hover),
            // 마우스가 다른 차트로 가면 그 차트가 새 위치를 알린다. 이 차트에서 벗어난 경우만 지운다.
            None if self
                .hover
                .is_some_and(|current| current.panel() == (kind_fan, panel)) =>
            {
                None
            }
            None => self.hover,
        };
        if next != self.hover {
            self.hover = next;
            cx.notify();
        }
    }

    fn toggle(&mut self, section: Section, cx: &mut Context<Self>) {
        match self.open.iter().position(|open| *open == section) {
            Some(ix) => {
                self.open.remove(ix);
            }
            None => self.open.push(section),
        }
        cx.notify();
    }

    fn groups(&self) -> Vec<&'static str> {
        match self.group {
            Some(group) => vec![group],
            None => GROUPS.to_vec(),
        }
    }

    fn keeps_group(&self, group: &str) -> bool {
        self.group.is_none_or(|selected| selected == group)
    }

    // ----- 화면 구역 -----

    fn render_daily(&self, report: &Report, cx: &mut Context<Self>) -> Div {
        let date = report.dates.get(self.date).cloned().unwrap_or_default();
        let days: Vec<_> = self
            .groups()
            .into_iter()
            .filter_map(|group| report.day(&date, group))
            .collect();
        let y_max = days
            .iter()
            .flat_map(|day| {
                day.actual
                    .iter()
                    .flatten()
                    .copied()
                    .chain(day.quantiles.iter().flatten().flatten().copied())
            })
            .fold(0.1_f64, f64::max)
            * 1.07;
        let view = cx.entity().downgrade();
        let mut panels = Vec::new();
        for (d, day) in days.iter().enumerate() {
            for (e, expert) in report.experts.iter().enumerate() {
                let panel = d * report.experts.len() + e;
                let hovered = match self.hover {
                    Some(Hover::Fan { panel: p, hour }) if p == panel => Some(hour),
                    _ => None,
                };
                let weak = view.clone();
                let chart = charts::fan(day, e, y_max, hovered, move |hour, _, cx| {
                    update_hover(
                        &weak,
                        true,
                        panel,
                        hour.map(|hour| Hover::Fan { panel, hour }),
                        cx,
                    )
                });
                let title = format!(
                    "{} · {} · {} MW",
                    day.group,
                    expert.label,
                    num(Some(day.capacity_mw), 2)
                );
                panels.push(chart_panel(title, 300., chart.into_any_element()));
            }
        }
        let group_button = |id: &'static str, label: &'static str, value: Option<&'static str>| {
            toggle_button(id, label, self.group == value).on_click(cx.listener(
                move |this, _: &ClickEvent, _, cx| {
                    this.group = value;
                    this.hover = None;
                    cx.notify();
                },
            ))
        };
        let last = report.dates.len().saturating_sub(1);
        div()
            .child(heading("같은 날짜·그룹의 예측구간"))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap_3()
                    .my_2()
                    .child(
                        small_button("previous", "← 이전")
                            .when(self.date == 0, |button| button.opacity(0.4))
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| this.move_date(-1, cx))),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child("대상일")
                            .child(
                                small_button(
                                    "date",
                                    format!(
                                        "{} ▾",
                                        report.dates.get(self.date).map_or("", String::as_str)
                                    ),
                                )
                                .on_click(cx.listener(
                                    |this, event: &ClickEvent, window, cx| {
                                        let at = event.position();
                                        this.open_month_menu(
                                            point(at.x - px(12.), at.y + px(12.)),
                                            window,
                                            cx,
                                        )
                                    },
                                )),
                            ),
                    )
                    .child(
                        small_button("next", "다음 →")
                            .when(self.date >= last, |button| button.opacity(0.4))
                            .on_click(cx.listener(|this, _: &ClickEvent, _, cx| this.move_date(1, cx))),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .child(div().mr_1().child("그룹"))
                            .child(group_button("group-all", "G1 · G2", None))
                            .child(group_button("group-g1", "G1", Some("G1")))
                            .child(group_button("group-g2", "G2", Some("G2"))),
                    ),
            )
            .child(
                muted("실선 중앙값 · 점선 실제 발전량 · 안쪽부터 50% · 80% · 90% 예측구간 · 대상일을 누르면 달과 날짜를 골라 이동"),
            )
            .child(div().flex().flex_wrap().children(panels))
            .child(div().min_h(px(24.)).text_color(theme::text()).child(self.fan_readout(report)))
    }

    fn fan_readout(&self, report: &Report) -> SharedString {
        let Some(Hover::Fan { panel, hour }) = self.hover else {
            return "차트에 마우스를 올리면 그 시각의 값을 표시합니다.".into();
        };
        let date = report.dates.get(self.date).cloned().unwrap_or_default();
        let groups = self.groups();
        let Some(day) = groups
            .get(panel / report.experts.len())
            .and_then(|group| report.day(&date, group))
        else {
            return SharedString::default();
        };
        let predictions: Vec<String> = report
            .experts
            .iter()
            .enumerate()
            .map(|(ix, expert)| {
                let q = day.quantiles[ix][hour];
                format!(
                    "{} 중앙값 {}, 80% [{}, {}]",
                    expert.label,
                    num(Some(q[MEDIAN]), 3),
                    num(Some(q[1]), 3),
                    num(Some(q[5]), 3)
                )
            })
            .collect();
        let unscored = if day.valid.get(hour).copied().unwrap_or(false) {
            ""
        } else {
            " · 실적 미채점"
        };
        format!(
            "{} {} {}시 · 실제 {} · {} MWh{unscored}",
            day.date,
            day.group,
            hour + 1,
            num(day.actual.get(hour).copied().flatten(), 3),
            predictions.join(" · ")
        )
        .into()
    }

    fn render_tables(&self, report: &Report, cx: &mut Context<Self>) -> Div {
        let mut seasons = vec!["all".to_string()];
        seasons.extend(report.seasons.iter().cloned());
        let season_buttons = seasons.into_iter().map(|season| {
            let selected = self.season == season;
            let id = SharedString::from(format!("season-{season}"));
            let label = season_name(&season).to_string();
            toggle_button(id, label, selected).on_click(cx.listener(
                move |this, _: &ClickEvent, _, cx| {
                    this.season = season.clone();
                    cx.notify();
                },
            ))
        });
        let summary: Vec<Vec<Cell>> = report
            .summary
            .iter()
            .filter(|row| row.season == self.season && self.keeps_group(&row.group))
            .map(|row| {
                vec![
                    Cell::new(format!("{} · {}", report.label(&row.expert), row.group)),
                    Cell::new(int(row.days)).right(),
                    Cell::new(int(row.n)).right(),
                    Cell::new(num(row.median_bias_kwh, 2)).right(),
                    Cell::new(num(row.mean_bias_kwh, 2)).right(),
                    Cell::new(num(row.pinball_kwh, 2)).right(),
                    Cell::new(num(row.w_krw.map(|w| w / 1e6), 3)).right(),
                    Cell::new(pct(row.crossing_rate)).right(),
                ]
            })
            .collect();
        let intervals: Vec<Vec<Cell>> = report
            .intervals
            .iter()
            .filter(|row| row.season == self.season && self.keeps_group(&row.group))
            .map(|row| {
                vec![
                    Cell::new(format!("{} · {}", report.label(&row.expert), row.group)),
                    Cell::new(format!("{}%", row.level)).right(),
                    Cell::new(pct(row.coverage)).right(),
                    Cell::new(pct(row.width_norm)).right(),
                    Cell::new(pct(row.below)).right(),
                    Cell::new(pct(row.above)).right(),
                    Cell::new(pct(row.endpoint_tie_rate)).right(),
                ]
            })
            .collect();
        div()
            .child(heading("계절별 비교표"))
            .child(muted(
                "편향 = 예측−실제(양수: 과대예측). 편향·pinball·구간 지표는 두 그룹 공통 주간 유효 시각, W는 전체 유효 시각 기준. 구간 폭은 설비용량으로 정규화합니다. 그룹 선택은 위의 날짜 구역과 같습니다.",
            ))
            .child(
                div()
                    .flex()
                    .flex_wrap()
                    .items_center()
                    .gap_1()
                    .my_2()
                    .child(div().mr_1().child("계절"))
                    .children(season_buttons),
            )
            .child(data_table("summary", "중앙값과 분위수 성능", &SUMMARY, 1, summary))
            .child(data_table("intervals", "예측구간의 적중률과 폭", &INTERVALS, 1, intervals))
    }

    fn render_behavior(&self, report: &Report, cx: &mut Context<Self>) -> Div {
        let mut metric_panels = Vec::new();
        for group in GROUPS {
            metric_panels.push(chart_panel(
                format!("{group} · 중앙값 편향"),
                240.,
                charts::season_metric(report, group, SeasonMetric::Bias).into_any_element(),
            ));
            metric_panels.push(chart_panel(
                format!("{group} · 평균 pinball"),
                240.,
                charts::season_metric(report, group, SeasonMetric::Pinball).into_any_element(),
            ));
        }
        let mut interval_panels = Vec::new();
        for width in [false, true] {
            for level in [50, 80, 90] {
                let title = if width {
                    format!("{level}% 구간 · 평균 폭")
                } else {
                    format!("{level}% 구간 · coverage")
                };
                interval_panels.push(
                    chart_panel(
                        title,
                        220.,
                        charts::interval_metric(report, level, width).into_any_element(),
                    )
                    .w(relative(1. / 3.)),
                );
            }
        }
        div()
            .child(heading("계절과 시간대의 behavior"))
            .child(subheading(
                "계절별 중앙값 편향과 pinball (공통 주간 유효 시각, 용량 대비 %)",
            ))
            .child(self.legend(report, false))
            .child(div().flex().flex_wrap().children(metric_panels))
            .child(subheading(
                "50 · 80 · 90% 구간의 계절별 coverage와 용량 대비 폭 (점선 기준선: 명목 coverage)",
            ))
            .child(self.legend(report, true))
            .child(div().flex().flex_wrap().children(interval_panels))
            .child(self.render_reliability(report, cx))
            .child(self.render_hourly(report, cx))
    }

    fn legend(&self, report: &Report, by_group: bool) -> Div {
        div()
            .flex()
            .flex_wrap()
            .gap_4()
            .my_1()
            .text_xs()
            .text_color(theme::text_muted())
            .children(report.experts.iter().enumerate().map(|(ix, expert)| {
                div()
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(div().w(px(14.)).h(px(3.)).bg(charts::expert_color(ix)))
                    .child(expert.label.clone())
            }))
            .when(by_group, |legend| legend.child("실선 G1 · 점선 G2"))
    }

    fn section_header(
        &self,
        section: Section,
        title: &'static str,
        cx: &mut Context<Self>,
    ) -> Stateful<Div> {
        let open = self.open.contains(&section);
        div()
            .id(SharedString::from(format!("section-{section:?}")))
            .mt_4()
            .py_2()
            .flex()
            .items_center()
            .gap_2()
            .cursor_pointer()
            .text_base()
            .text_color(theme::text())
            .hover(|style| style.text_color(theme::accent()))
            .child(if open { "▾" } else { "▸" })
            .child(title)
            .on_click(cx.listener(move |this, _: &ClickEvent, _, cx| this.toggle(section, cx)))
    }

    fn render_reliability(&self, report: &Report, cx: &mut Context<Self>) -> Div {
        let header = self.section_header(Section::Reliability, "분위수별 적중률 · 경계값 일치", cx);
        if !self.open.contains(&Section::Reliability) {
            return div().child(header);
        }
        let columns = report.seasons.len().max(1) as f32;
        let mut panels = Vec::new();
        for group in GROUPS {
            for season in &report.seasons {
                panels.push(
                    chart_panel(
                        format!("{group} · {}", season_name(season)),
                        230.,
                        charts::reliability(report, group, season).into_any_element(),
                    )
                    .w(relative(1. / columns)),
                );
            }
        }
        div()
            .child(header)
            .child(muted(
                "색 띠는 P(Y<q)와 P(Y≤q) 사이입니다. 발전량 0 등 경계값의 확률 질량을 함께 표시합니다. 점선은 calibration 기준선입니다.",
            ))
            .child(self.legend(report, false))
            .child(div().flex().flex_wrap().children(panels))
    }

    fn render_hourly(&self, report: &Report, cx: &mut Context<Self>) -> Div {
        let header = self.section_header(Section::Hourly, "계절 × 시간대 편향", cx);
        if !self.open.contains(&Section::Hourly) {
            return div().child(header);
        }
        let bound = charts::hourly_bound(report);
        let view = cx.entity().downgrade();
        let mut panels = Vec::new();
        for expert in &report.experts {
            for group in GROUPS {
                let panel = panels.len();
                let hovered = match self.hover {
                    Some(Hover::Heat { panel: p, cell }) if p == panel => Some(cell),
                    _ => None,
                };
                let weak = view.clone();
                let heatmap = charts::hourly(
                    report,
                    &expert.id,
                    group,
                    bound,
                    hovered,
                    move |cell, _, cx| {
                        update_hover(
                            &weak,
                            false,
                            panel,
                            cell.map(|cell| Hover::Heat { panel, cell }),
                            cx,
                        )
                    },
                );
                panels.push(chart_panel(
                    format!("{} · {group}", expert.label),
                    40. + 34. * report.seasons.len() as f32,
                    heatmap.into_any_element(),
                ));
            }
        }
        div()
            .child(header)
            .child(muted("중앙값 편향 / 용량(%). 빨강은 과대예측, 파랑은 과소예측이고 빈 칸은 공통 주간 표본이 없는 시각입니다."))
            .child(div().flex().flex_wrap().children(panels))
            .child(div().min_h(px(24.)).text_color(theme::text()).child(self.heat_readout(report)))
    }

    fn heat_readout(&self, report: &Report) -> SharedString {
        let Some(Hover::Heat {
            panel,
            cell: (season, hour),
        }) = self.hover
        else {
            return "칸에 마우스를 올리면 값을 표시합니다.".into();
        };
        let expert = &report.experts[panel / GROUPS.len()];
        let group = GROUPS[panel % GROUPS.len()];
        let Some(season_id) = report.seasons.get(season) else {
            return SharedString::default();
        };
        let value = charts::hourly_values(report, &expert.id, group)[season][hour];
        let value = match value {
            Some(value) => format!("중앙값 편향 {value:+.2}%"),
            None => "공통 주간 표본 없음".into(),
        };
        format!(
            "{} · {group} · {} · {}시 · {value}",
            expert.label,
            season_name(season_id),
            hour + 1
        )
        .into()
    }

    fn render_regimes(&self, report: &Report, cx: &mut Context<Self>) -> Div {
        if report.regimes.is_empty() {
            return div();
        }
        let header = self.section_header(Section::Regimes, "예보 운량·제공사 불일치별 비교", cx);
        if !self.open.contains(&Section::Regimes) {
            return div().child(header);
        }
        let edges = |axis: &str| -> String {
            report.manifest["regime_edges"][axis]
                .as_array()
                .map(|edges| {
                    edges
                        .iter()
                        .map(|edge| num(edge.as_f64(), 2))
                        .collect::<Vec<_>>()
                        .join(" / ")
                })
                .unwrap_or_default()
        };
        let rows: Vec<Vec<Cell>> = report
            .regimes
            .iter()
            .filter(|row| self.keeps_group(&row.group))
            .map(|row| {
                let axis = if row.axis == "cloud" {
                    "운량"
                } else {
                    "불일치"
                };
                vec![
                    Cell::new(format!("{axis} · {}구간", row.bin + 1)),
                    Cell::new(format!("{} · {}", report.label(&row.expert), row.group)),
                    Cell::new(int(row.n)).right(),
                    Cell::new(num(row.median_bias_kwh, 2)).right(),
                    Cell::new(pct(row.coverage80)).right(),
                    Cell::new(pct(row.width80_norm)).right(),
                    Cell::new(num(row.pinball_kwh, 2)).right(),
                ]
            })
            .collect();
        div()
            .child(header)
            .child(muted(format!(
                "17시 발행 예보의 그룹 용량 가중 운량과 WA–WB 차이. 학습 예보에서 고정한 경계: 운량 {}%, 불일치 {}%p. 동일값은 같은 구간에 둡니다.",
                edges("cloud"),
                edges("disagreement")
            )))
            .child(data_table("regimes", "", &REGIMES, 2, rows))
    }

    fn render_report(&self, report: &Report, cx: &mut Context<Self>) -> Div {
        let m = &report.manifest;
        let seeds = m["seeds"]
            .as_array()
            .map(|seeds| {
                seeds
                    .iter()
                    .map(|seed| seed.to_string())
                    .collect::<Vec<_>>()
                    .join("·")
            })
            .unwrap_or_default();
        let site_aux_ratio = m["site_aux_ratio"]
            .as_f64()
            .map(|r| format!(" · DL site_aux_ratio={r}"))
            .unwrap_or_default();
        let text = |key: &str, fallback: &str| m[key].as_str().unwrap_or(fallback).to_string();
        let meta = format!(
            "{} ~ {} · {}개소 고정 cohort · {} 배정 · seeds {} · 저장 artifact 평가{site_aux_ratio}",
            report.dates.first().cloned().unwrap_or_default(),
            report.dates.last().cloned().unwrap_or_default(),
            m["cohort_size"]
                .as_u64()
                .map_or("—".into(), |n| n.to_string()),
            text("partition", "고정"),
            seeds,
        );
        let provenance = format!(
            "Source: {} | 추론 key: {} | 학습 cutoff: {} | 주간 공통 유효 시각: {} | 단위: 곡선 MWh(1시간 발전량), 원본 kWh | 데이터: {}",
            text("source_instance", "저장 artifact"),
            text("key", "—"),
            text("cutoff", "—"),
            num(m["valid_daylight_hours"].as_f64().or(Some(0.)), 0),
            report.path.display()
        );
        div()
            .child(
                div()
                    .text_2xl()
                    .text_color(theme::text())
                    .child(format!("{} · 확률예측 비교", report.instance())),
            )
            .child(muted(meta))
            .child(self.render_daily(report, cx))
            .child(self.render_tables(report, cx))
            .child(self.render_behavior(report, cx))
            .child(self.render_regimes(report, cx))
            .child(
                div()
                    .mt_6()
                    .pt_2()
                    .border_t_1()
                    .border_color(theme::border())
                    .text_xs()
                    .text_color(theme::text_muted())
                    .child(provenance),
            )
    }
}

fn update_hover(
    view: &WeakEntity<Explorer>,
    kind_fan: bool,
    panel: usize,
    hover: Option<Hover>,
    cx: &mut gpui::App,
) {
    view.update(cx, |this, cx| this.set_hover(kind_fan, panel, hover, cx))
        .ok();
}

fn heading(text: &'static str) -> Div {
    div()
        .mt_6()
        .mb_1()
        .text_lg()
        .text_color(theme::text())
        .child(text)
}

fn subheading(text: &'static str) -> Div {
    div().mt_3().text_color(theme::text()).child(text)
}

fn muted(text: impl Into<SharedString>) -> Div {
    div()
        .text_xs()
        .text_color(theme::text_muted())
        .child(text.into())
}

/// 제목과 차트 하나. 기본 너비는 화면의 절반이다.
fn chart_panel(title: impl Into<SharedString>, height: f32, chart: AnyElement) -> Div {
    div()
        .w(relative(0.5))
        .min_w(px(320.))
        .p_2()
        .flex()
        .flex_col()
        .gap_1()
        .child(div().text_color(theme::text()).child(title.into()))
        .child(div().h(px(height)).w_full().text_xs().child(chart))
}

/// 표 하나. 앞의 `text_columns`개 칸은 글자, 나머지는 수치로 보고 머리글을 오른쪽에 맞춘다.
fn data_table(
    name: &'static str,
    caption: &'static str,
    columns: &[ColumnSpec],
    text_columns: usize,
    rows: Vec<Vec<Cell>>,
) -> Div {
    let empty = rows.is_empty();
    div()
        .my_2()
        .when(!caption.is_empty(), |table| {
            table.child(div().py_1().text_color(theme::text()).child(caption))
        })
        .child(
            div()
                .border_1()
                .border_color(theme::border())
                .text_xs()
                .child(table::header_aligned(columns, |ix| ix >= text_columns))
                .children(
                    rows.into_iter()
                        .enumerate()
                        .map(|(ix, cells)| table::row(name, ix, columns, cells)),
                )
                .when(empty, |table| {
                    table.child(
                        div()
                            .p_2()
                            .text_color(theme::text_muted())
                            .child("해당하는 행이 없습니다."),
                    )
                }),
        )
}

impl Render for Explorer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let title = match &self.report {
            Some(report) => format!("{} · 확률예측 비교", report.instance()),
            None => "확률예측 비교".to_string(),
        };
        if self.title.as_deref() != Some(title.as_str()) {
            window.set_window_title(&title);
            self.title = Some(title);
        }
        let body = match (&self.status, &self.report) {
            (Status::Ready, Some(report)) => self.render_report(report, cx),
            (Status::Loading(path), _) => muted(format!("{} 읽는 중", path.display())).p_4(),
            (Status::Failed(message), _) => div().p_4().text_color(theme::error()).child(message.clone()),
            _ => muted(
                "결과 폴더 열기(Ctrl+O)로 vppcontract probabilistic이 만든 결과 폴더(explorer-data.json이 있는 폴더)를 엽니다.",
            )
            .p_4(),
        };
        div()
            .id("explorer")
            .key_context("Explorer")
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(Self::open_folder))
            .size_full()
            .overflow_y_scroll()
            .bg(theme::background())
            .font_family(theme::ui_font())
            .text_sm()
            .text_color(theme::text_muted())
            .p_4()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        small_button("open-folder", "결과 폴더 열기").on_click(cx.listener(
                            |this, _: &ClickEvent, window, cx| {
                                this.open_folder(&OpenFolder, window, cx)
                            },
                        )),
                    )
                    .child(div().flex_1())
                    .child(
                        small_button(
                            "theme-mode",
                            match theme::mode() {
                                theme::Mode::Dark => "밝은 화면",
                                theme::Mode::Light => "어두운 화면",
                            },
                        )
                        .on_click(cx.listener(
                            |this, _: &ClickEvent, _, cx| {
                                this.follow_system = false;
                                theme::set_mode(theme::mode().toggled());
                                cx.notify();
                            },
                        )),
                    ),
            )
            .child(body)
            .children(
                self.date_menu
                    .as_ref()
                    .map(|(menu, position, _)| menu::overlay(menu, *position)),
            )
    }
}
