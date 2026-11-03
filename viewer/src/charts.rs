//! 결과 데이터를 차트로 바꾼다. 값은 파일에 있는 그대로 쓰고 축 범위만 정한다.

use gpui::{App, Hsla, Window, px, rgb};
use viewer_ui::chart::{Axis, Band, Line, LineChart, Rule};
use viewer_ui::heatmap::{DivergingScale, Heatmap};
use viewer_ui::theme;

use crate::data::{DayGroup, GROUPS, MEDIAN, Report, season_name};

/// 분위수 띠의 아래·위 순번과 불투명도. 안쪽부터 50%·80%·90% 예측구간이다.
const BANDS: [(usize, usize, f32); 3] = [(0, 6, 0.10), (1, 5, 0.14), (2, 4, 0.20)];

/// 비교 expert의 고정 색. 밝은 화면에서는 더 진한 색을 쓴다.
pub fn expert_color(ix: usize) -> Hsla {
    match ix % 6 {
        0 => theme::pick(0x65b5e7, 0x2373b8).into(),
        1 => theme::pick(0xffa475, 0xd9601c).into(),
        2 => theme::pick(0x77cda2, 0x24784c).into(),
        3 => theme::pick(0xc1a3f5, 0x8250b8).into(),
        4 => theme::pick(0xe6ca68, 0x8a6915).into(),
        _ => theme::text().into(),
    }
}

fn faded(color: Hsla, alpha: f32) -> Hsla {
    Hsla { a: alpha, ..color }
}

fn hours() -> Vec<f64> {
    (1..=24).map(f64::from).collect()
}

/// 값들을 담는 축 범위. 범위의 10%를 위아래로 더한다.
fn padded(values: impl IntoIterator<Item = f64>, include: &[f64]) -> (f64, f64) {
    let (mut low, mut high) = include
        .iter()
        .copied()
        .chain(values)
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    if !low.is_finite() {
        (low, high) = (0., 1.);
    }
    let pad = ((high - low) * 0.1).max(1e-6);
    (low - pad, high + pad)
}

/// 날짜·그룹·expert 하나의 24시간 예측구간. 실선은 중앙값, 점선은 실제 발전량이다.
pub fn fan(
    day: &DayGroup,
    expert: usize,
    y_max: f64,
    hovered: Option<usize>,
    on_hover: impl Fn(Option<usize>, &mut Window, &mut App) + 'static,
) -> LineChart {
    let color = expert_color(expert);
    let q = &day.quantiles[expert];
    let column = |level: usize| q.iter().map(|hour| Some(hour[level])).collect::<Vec<_>>();
    let mut chart = LineChart::new(
        Axis::new(1., 24.)
            .ticks([1., 6., 12., 18., 24.].map(|h| (h, h.to_string().into())))
            .title("종료 시각 (KST)"),
        Axis::new(0., y_max).nice_ticks(4).title("MWh"),
    );
    for (lower, upper, alpha) in BANDS {
        chart = chart.layer(Band::new(
            hours(),
            column(lower),
            column(upper),
            faded(color, alpha),
        ));
    }
    chart
        .layer(Line::new(hours(), column(MEDIAN), color))
        .layer(
            Line::new(hours(), day.actual.clone(), theme::text().into())
                .width(px(1.6))
                .dashed(px(4.), px(3.)),
        )
        .on_hover(hours(), hovered, on_hover)
}

/// 계절별 비교 차트의 지표.
#[derive(Clone, Copy)]
pub enum SeasonMetric {
    /// 중앙값 편향 / 용량(%).
    Bias,
    /// 평균 pinball / 용량(%).
    Pinball,
}

fn season_axis(report: &Report) -> Axis {
    let count = report.seasons.len().max(1);
    Axis::new(-0.4, count as f64 - 0.6).ticks(
        report
            .seasons
            .iter()
            .enumerate()
            .map(|(ix, season)| (ix as f64, season_name(season).to_string().into())),
    )
}

/// 한 그룹의 계절별 중앙값 편향 또는 pinball. expert마다 선 하나.
pub fn season_metric(report: &Report, group: &str, metric: SeasonMetric) -> LineChart {
    let x: Vec<f64> = (0..report.seasons.len()).map(|ix| ix as f64).collect();
    let lines: Vec<(usize, Vec<Option<f64>>)> = report
        .experts
        .iter()
        .enumerate()
        .map(|(ix, expert)| {
            let values = report
                .seasons
                .iter()
                .map(|season| {
                    report
                        .summary
                        .iter()
                        .find(|row| {
                            row.expert == expert.id && row.group == group && &row.season == season
                        })
                        .and_then(|row| match metric {
                            SeasonMetric::Bias => row.median_bias_norm,
                            SeasonMetric::Pinball => row.pinball_norm,
                        })
                        .map(|value| value * 100.)
                })
                .collect();
            (ix, values)
        })
        .collect();
    let (low, high) = padded(
        lines
            .iter()
            .flat_map(|(_, values)| values.iter().flatten().copied()),
        &[0.],
    );
    let title = match metric {
        SeasonMetric::Bias => "중앙값 편향 / 용량 (%)",
        SeasonMetric::Pinball => "평균 pinball / 용량 (%)",
    };
    let mut chart = LineChart::new(
        season_axis(report),
        Axis::new(low, high).nice_ticks(4).title(title),
    );
    if matches!(metric, SeasonMetric::Bias) {
        chart = chart.layer(Rule::new(
            (-1., 0.),
            (x.len() as f64, 0.),
            theme::text_muted().into(),
        ));
    }
    for (ix, values) in lines {
        chart = chart.layer(Line::new(x.clone(), values, expert_color(ix)).markers());
    }
    chart
}

/// 한 구간 수준의 계절별 coverage(명목값 기준선 포함) 또는 용량 대비 폭. 실선 G1, 점선 G2.
pub fn interval_metric(report: &Report, level: u32, width: bool) -> LineChart {
    let x: Vec<f64> = (0..report.seasons.len()).map(|ix| ix as f64).collect();
    let mut lines = Vec::new();
    for (ix, expert) in report.experts.iter().enumerate() {
        for (g, group) in GROUPS.into_iter().enumerate() {
            let values: Vec<Option<f64>> = report
                .seasons
                .iter()
                .map(|season| {
                    report
                        .intervals
                        .iter()
                        .find(|row| {
                            row.expert == expert.id
                                && row.group == group
                                && &row.season == season
                                && row.level == level
                        })
                        .and_then(|row| if width { row.width_norm } else { row.coverage })
                        .map(|value| value * 100.)
                })
                .collect();
            let line = Line::new(x.clone(), values, expert_color(ix)).markers();
            lines.push(if g == 1 {
                line.dashed(px(5.), px(3.))
            } else {
                line
            });
        }
    }
    let nominal = f64::from(level);
    let include: &[f64] = if width { &[0.] } else { &[nominal] };
    let (low, high) = padded(
        lines
            .iter()
            .flat_map(|line| line.y.iter().flatten().copied()),
        include,
    );
    let title = if width {
        format!("{level}% 구간 평균 폭 / 용량 (%)")
    } else {
        format!("{level}% 구간 실제 coverage (%)")
    };
    let mut chart = LineChart::new(
        season_axis(report),
        Axis::new(low, high).nice_ticks(4).title(title),
    );
    if !width {
        chart = chart.layer(
            Rule::new(
                (-1., nominal),
                (x.len() as f64, nominal),
                theme::text_muted().into(),
            )
            .dashed(px(2.), px(3.)),
        );
    }
    for line in lines {
        chart = chart.layer(line);
    }
    chart
}

/// 한 그룹·계절의 분위수 신뢰도. 선은 P(Y≤q), 띠는 P(Y<q)와 P(Y≤q) 사이의 경계값 확률 질량이다.
pub fn reliability(report: &Report, group: &str, season: &str) -> LineChart {
    let mut chart = LineChart::new(
        Axis::new(0., 1.).nice_ticks(4).title("분위수 수준"),
        Axis::new(0., 1.).nice_ticks(4).title("P(Y<q) ~ P(Y≤q)"),
    )
    .layer(Rule::new((0., 0.), (1., 1.), theme::text_muted().into()).dashed(px(2.), px(3.)));
    for (ix, expert) in report.experts.iter().enumerate() {
        let mut rows: Vec<_> = report
            .reliability
            .iter()
            .filter(|row| row.expert == expert.id && row.group == group && row.season == season)
            .collect();
        rows.sort_by(|a, b| a.alpha.total_cmp(&b.alpha));
        let alpha: Vec<f64> = rows.iter().map(|row| row.alpha).collect();
        let strict = rows.iter().map(|row| row.strict).collect();
        let inclusive: Vec<Option<f64>> = rows.iter().map(|row| row.inclusive).collect();
        let color = expert_color(ix);
        chart = chart
            .layer(Band::new(
                alpha.clone(),
                strict,
                inclusive.clone(),
                faded(color, 0.2),
            ))
            .layer(Line::new(alpha, inclusive, color).width(px(1.5)));
    }
    chart
}

/// 모든 expert·그룹에 같은 색 척도를 쓰도록, 시간대 편향의 절댓값 최대(%)를 구한다.
pub fn hourly_bound(report: &Report) -> f64 {
    report
        .hourly
        .iter()
        .filter_map(|row| row.median_bias_norm)
        .fold(0.01_f64, |bound, value| bound.max(value.abs() * 100.))
}

/// 시간대 편향 값. `values[계절][시각 0..24]`, 단위는 용량 대비 %.
pub fn hourly_values(report: &Report, expert: &str, group: &str) -> Vec<Vec<Option<f64>>> {
    report
        .seasons
        .iter()
        .map(|season| {
            (1..=24)
                .map(|hour| {
                    report
                        .hourly
                        .iter()
                        .find(|row| {
                            row.expert == expert
                                && row.group == group
                                && &row.season == season
                                && row.hour == hour
                        })
                        .and_then(|row| row.median_bias_norm)
                        .map(|value| value * 100.)
                })
                .collect()
        })
        .collect()
}

/// expert·그룹 하나의 계절 × 시간대 중앙값 편향. 파랑은 과소, 빨강은 과대예측이다.
pub fn hourly(
    report: &Report,
    expert: &str,
    group: &str,
    bound: f64,
    hovered: Option<(usize, usize)>,
    on_hover: impl Fn(Option<(usize, usize)>, &mut Window, &mut App) + 'static,
) -> Heatmap {
    let scale = DivergingScale {
        bound,
        negative: rgb(0x2166ac),
        middle: theme::pick(0x3d4045, 0xdcdfe4),
        positive: rgb(0xb2182b),
    };
    Heatmap::new(
        report
            .seasons
            .iter()
            .map(|season| season_name(season).to_string().into())
            .collect(),
        (1..=24).map(|hour| hour.to_string().into()).collect(),
        hourly_values(report, expert, group),
        scale,
    )
    .column_ticks(vec![0, 5, 11, 17, 23])
    .format(|value| format!("{value:+.1}%"))
    .on_hover(hovered, on_hover)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pads_ranges_and_keeps_reference_values() {
        let (low, high) = padded([1., 3.], &[0.]);
        assert!(low < 0. && high > 3.);
        let (low, high) = padded([f64::NAN], &[]);
        assert!(low < 0. && high > 1.);
        let (low, high) = padded([80., 82.], &[80.]);
        assert!(low < 80. && high > 82.);
    }
}
