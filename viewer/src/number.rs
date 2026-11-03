//! 표와 안내 문구의 수 표시. 세 자리마다 쉼표를 넣고 소수 자릿수를 고정한다.

/// 소수 `decimals`자리로 반올림하고 정수부에 쉼표를 넣는다. 값이 없거나 유한하지 않으면 `—`다.
pub fn num(value: Option<f64>, decimals: usize) -> String {
    let Some(value) = value.filter(|value| value.is_finite()) else {
        return "—".into();
    };
    let text = format!("{:.*}", decimals, value);
    let (sign, digits) = match text.strip_prefix('-') {
        // 반올림해서 0이 된 음수는 부호를 붙이지 않는다.
        Some(rest) if rest.chars().any(|c| c.is_ascii_digit() && c != '0') => ("-", rest),
        Some(rest) => ("", rest),
        None => ("", text.as_str()),
    };
    let (integer, fraction) = match digits.split_once('.') {
        Some((integer, fraction)) => (integer, Some(fraction)),
        None => (digits, None),
    };
    let mut grouped = String::with_capacity(integer.len() + integer.len() / 3);
    for (ix, digit) in integer.chars().enumerate() {
        if ix > 0 && (integer.len() - ix).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    match fraction {
        Some(fraction) => format!("{sign}{grouped}.{fraction}"),
        None => format!("{sign}{grouped}"),
    }
}

/// 비율을 소수 첫째 자리 백분율로 표시한다.
pub fn pct(value: Option<f64>) -> String {
    match value.filter(|value| value.is_finite()) {
        Some(value) => format!("{}%", num(Some(100. * value), 1)),
        None => "—".into(),
    }
}

/// 정수를 쉼표 없이 표시한다. 표의 일수·시각 수처럼 원래 값을 그대로 보일 때 쓴다.
pub fn int(value: Option<u64>) -> String {
    value.map_or("—".into(), |value| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_thousands_and_fixes_decimals() {
        assert_eq!(num(Some(-59.9512), 2), "-59.95");
        assert_eq!(num(Some(1234567.891), 2), "1,234,567.89");
        assert_eq!(num(Some(84.344451), 3), "84.344");
        assert_eq!(num(Some(999.999), 2), "1,000.00");
        assert_eq!(num(Some(-0.0004), 2), "0.00");
        assert_eq!(num(Some(12.0), 0), "12");
        assert_eq!(num(None, 2), "—");
        assert_eq!(num(Some(f64::NAN), 2), "—");
    }

    #[test]
    fn shows_percentages_with_one_decimal() {
        assert_eq!(pct(Some(0.4756)), "47.6%");
        assert_eq!(pct(Some(0.0)), "0.0%");
        assert_eq!(pct(None), "—");
    }
}
