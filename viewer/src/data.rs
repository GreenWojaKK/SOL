//! `vppcontract probabilistic`이 만든 `explorer-data.json`을 읽는다.
//!
//! 값은 Python 공통 평가가 계산한 그대로 쓰고, viewer는 다시 계산하지 않는다. 읽을 때 날짜·그룹 행의
//! 24시각·분위수 7개 형식과 표의 필수 항목을 확인한다.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// 결과 폴더 안의 viewer 데이터 파일 이름.
pub const DATA_FILE: &str = "explorer-data.json";
/// 분위수 7개(0.05·0.1·0.25·0.5·0.75·0.9·0.95)의 순서.
pub const LEVELS: [f64; 7] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
pub const MEDIAN: usize = 3;
pub const GROUPS: [&str; 2] = ["G1", "G2"];

#[derive(Clone, Debug)]
pub struct Expert {
    pub id: String,
    pub label: String,
}

/// 한 대상일·그룹의 24시각 실적과 expert별 분위수. 단위는 MWh(1시간 발전량)다.
#[derive(Clone, Debug)]
pub struct DayGroup {
    pub date: String,
    pub group: String,
    pub capacity_mw: f64,
    pub actual: Vec<Option<f64>>,
    /// 실적을 채점한 시각.
    pub valid: Vec<bool>,
    /// expert 순서대로 `[시각][분위수]`.
    pub quantiles: Vec<Vec<[f64; 7]>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SummaryRow {
    pub expert: String,
    pub group: String,
    pub season: String,
    pub days: Option<u64>,
    pub n: Option<u64>,
    pub median_bias_kwh: Option<f64>,
    pub mean_bias_kwh: Option<f64>,
    pub median_bias_norm: Option<f64>,
    pub pinball_kwh: Option<f64>,
    pub pinball_norm: Option<f64>,
    #[serde(rename = "W_krw")]
    pub w_krw: Option<f64>,
    pub crossing_rate: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IntervalRow {
    pub expert: String,
    pub group: String,
    pub season: String,
    pub level: u32,
    pub coverage: Option<f64>,
    pub width_norm: Option<f64>,
    pub below: Option<f64>,
    pub above: Option<f64>,
    pub endpoint_tie_rate: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReliabilityRow {
    pub expert: String,
    pub group: String,
    pub season: String,
    pub alpha: f64,
    pub strict: Option<f64>,
    pub inclusive: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct HourlyRow {
    pub expert: String,
    pub group: String,
    pub season: String,
    pub hour: u32,
    pub median_bias_norm: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RegimeRow {
    pub expert: String,
    pub group: String,
    pub axis: String,
    pub bin: u32,
    pub n: Option<u64>,
    pub median_bias_kwh: Option<f64>,
    pub coverage80: Option<f64>,
    pub width80_norm: Option<f64>,
    pub pinball_kwh: Option<f64>,
}

/// 결과 폴더 하나의 viewer 데이터.
#[derive(Debug)]
pub struct Report {
    pub path: PathBuf,
    pub manifest: Value,
    pub experts: Vec<Expert>,
    /// 유효 표본이 있는 계절. 파일에 적힌 순서다.
    pub seasons: Vec<String>,
    pub dates: Vec<String>,
    pub days: Vec<DayGroup>,
    index: HashMap<(String, String), usize>,
    pub summary: Vec<SummaryRow>,
    pub intervals: Vec<IntervalRow>,
    pub reliability: Vec<ReliabilityRow>,
    pub hourly: Vec<HourlyRow>,
    pub regimes: Vec<RegimeRow>,
}

impl Report {
    pub fn instance(&self) -> &str {
        self.manifest["instance"].as_str().unwrap_or("결과")
    }

    pub fn day(&self, date: &str, group: &str) -> Option<&DayGroup> {
        self.index
            .get(&(date.to_string(), group.to_string()))
            .map(|ix| &self.days[*ix])
    }

    pub fn label<'a>(&'a self, expert: &'a str) -> &'a str {
        self.experts
            .iter()
            .find(|candidate| candidate.id == expert)
            .map_or(expert, |candidate| candidate.label.as_str())
    }
}

/// 계절 ID의 표시 이름.
pub fn season_name(season: &str) -> &str {
    match season {
        "all" => "전체 기간",
        "spring" => "봄",
        "summer" => "여름",
        "autumn" => "가을",
        "winter" => "겨울",
        other => other,
    }
}

/// 결과 폴더 또는 `explorer-data.json` 경로를 받아 읽는다.
pub fn load(path: &Path) -> Result<Report, String> {
    let file = if path.is_dir() {
        path.join(DATA_FILE)
    } else {
        path.to_path_buf()
    };
    let text = std::fs::read_to_string(&file).map_err(|err| {
        format!(
            "{}을(를) 읽을 수 없습니다. vppcontract probabilistic으로 만든 결과 폴더를 고르세요. ({err})",
            file.display()
        )
    })?;
    let root: Value = serde_json::from_str(&text).map_err(|err| {
        format!(
            "{}의 JSON 형식이 올바르지 않습니다. ({err})",
            file.display()
        )
    })?;
    parse(root, file)
}

fn parse(root: Value, path: PathBuf) -> Result<Report, String> {
    let Some(experts) = root["experts"].as_object() else {
        return Err(
            "experts가 없는 이전 형식입니다. vppcontract probabilistic으로 결과를 다시 만드세요."
                .into(),
        );
    };
    let experts: Vec<Expert> = experts
        .iter()
        .map(|(id, meta)| Expert {
            id: id.clone(),
            label: meta["label"].as_str().unwrap_or(id).to_string(),
        })
        .collect();
    if experts.is_empty() {
        return Err("expert가 없습니다.".into());
    }
    let seasons = strings(&root["seasons"], "seasons")?;
    let levels: Vec<f64> = root["levels"]
        .as_array()
        .map(|levels| levels.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default();
    if levels != LEVELS {
        return Err(format!("분위수 순서가 {LEVELS:?}가 아닙니다: {levels:?}"));
    }

    let rows = root["rows"].as_array().ok_or("rows가 없습니다.")?;
    let mut dates: Vec<String> = Vec::new();
    let mut days = Vec::with_capacity(rows.len());
    let mut index = HashMap::new();
    for (ix, row) in rows.iter().enumerate() {
        let day = day_group(row, &experts).map_err(|err| format!("rows[{ix}]: {err}"))?;
        if dates.last() != Some(&day.date) {
            dates.push(day.date.clone());
        }
        index.insert((day.date.clone(), day.group.clone()), ix);
        days.push(day);
    }

    let tables = &root["tables"];
    Ok(Report {
        path,
        manifest: root["manifest"].clone(),
        experts,
        seasons,
        dates,
        days,
        index,
        summary: table(tables, "seasonal_summary")?,
        intervals: table(tables, "intervals")?,
        reliability: table(tables, "reliability")?,
        hourly: table(tables, "hourly_bias")?,
        regimes: table(tables, "weather_regimes")?,
    })
}

fn strings(value: &Value, name: &str) -> Result<Vec<String>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{name}이(가) 없습니다."))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .ok_or_else(|| format!("{name}의 항목이 문자열이 아닙니다."))
        })
        .collect()
}

fn table<T: DeserializeOwned>(tables: &Value, name: &str) -> Result<Vec<T>, String> {
    match &tables[name] {
        Value::Null => Ok(Vec::new()),
        value => serde_json::from_value(value.clone())
            .map_err(|err| format!("tables.{name}의 형식이 올바르지 않습니다. ({err})")),
    }
}

fn day_group(row: &Value, experts: &[Expert]) -> Result<DayGroup, String> {
    let text = |key: &str| {
        row[key]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("{key}가 없습니다."))
    };
    let hours = |key: &str| -> Result<&Vec<Value>, String> {
        let values = row[key]
            .as_array()
            .ok_or_else(|| format!("{key}가 없습니다."))?;
        if values.len() != 24 {
            return Err(format!("{key}는 24시각이어야 합니다."));
        }
        Ok(values)
    };
    let actual = hours("actual")?.iter().map(Value::as_f64).collect();
    let valid = hours("valid")?
        .iter()
        .map(|value| value.as_bool().unwrap_or(false))
        .collect();
    let quantiles = experts
        .iter()
        .map(|expert| {
            hours(&expert.id)?
                .iter()
                .map(|hour| {
                    let values: Vec<f64> = hour
                        .as_array()
                        .map(|values| values.iter().filter_map(Value::as_f64).collect())
                        .unwrap_or_default();
                    <[f64; 7]>::try_from(values)
                        .map_err(|_| format!("{}의 시각마다 분위수 7개가 필요합니다.", expert.id))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DayGroup {
        date: text("date")?,
        group: text("group")?,
        capacity_mw: row["capacity_mw"].as_f64().unwrap_or_default(),
        actual,
        valid,
        quantiles,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn sample() -> Value {
        let hours = |value: f64| vec![json!(value); 24];
        let quantiles = vec![json!([0., 1., 2., 3., 4., 5., 6.]); 24];
        json!({
            "schema_version": 2,
            "manifest": {"instance": "B"},
            "experts": {"zeta": {"label": "Z model"}, "alpha": {"label": "A model"}},
            "seasons": ["winter"],
            "levels": LEVELS,
            "rows": [
                {"date": "2024-12-25", "group": "G1", "capacity_mw": 0.01, "actual": hours(1.),
                 "valid": vec![json!(true); 24], "daylight": vec![json!(true); 24],
                 "zeta": quantiles, "alpha": quantiles},
            ],
            "tables": {
                "seasonal_summary": [{"expert": "zeta", "seed": "ensemble", "group": "G1", "season": "all",
                    "n": 3, "days": 2, "median_bias_kwh": 0.5, "mean_bias_kwh": null, "median_bias_norm": 0.05,
                    "pinball_kwh": 0.1, "pinball_norm": 0.01, "W_krw": 1.5, "crossing_rate": 0.0}],
                "intervals": [], "reliability": [], "hourly_bias": [], "weather_regimes": []
            }
        })
    }

    #[test]
    fn keeps_expert_order_and_indexes_days() {
        let report = parse(sample(), PathBuf::from("x")).unwrap();
        let ids: Vec<&str> = report.experts.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["zeta", "alpha"]);
        assert_eq!(report.instance(), "B");
        assert_eq!(report.dates, ["2024-12-25"]);
        let day = report.day("2024-12-25", "G1").unwrap();
        assert_eq!(day.quantiles[1][23][MEDIAN], 3.);
        assert!(report.day("2024-12-25", "G2").is_none());
        assert_eq!(report.summary[0].w_krw, Some(1.5));
        assert_eq!(report.summary[0].mean_bias_kwh, None);
    }

    #[test]
    fn rejects_old_payloads_and_short_rows() {
        let mut old = sample();
        old.as_object_mut().unwrap().remove("experts");
        assert!(
            parse(old, PathBuf::from("x"))
                .unwrap_err()
                .contains("이전 형식")
        );
        let mut short = sample();
        short["rows"][0]["actual"] = json!([1.0]);
        assert!(
            parse(short, PathBuf::from("x"))
                .unwrap_err()
                .contains("24시각")
        );
    }
}
