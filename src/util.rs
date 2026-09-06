use subtle::ConstantTimeEq;

/// Constant-time comparison of an attacker-supplied token against the expected
/// secret. Prevents a timing side-channel that a byte-by-byte `==` would leak
/// (early-exit on the first mismatching byte lets an attacker recover the secret
/// one byte at a time). The length of the two strings is not itself secret, so
/// leaking it via the short-circuit in `ConstantTimeEq for [u8]` is acceptable.
pub fn constant_time_str_eq(provided: &str, expected: &str) -> bool {
    provided.as_bytes().ct_eq(expected.as_bytes()).into()
}

/// Whether a bind host refers only to the local loopback interface. Used to
/// decide whether exposing an unauthenticated HTTP surface is safe. Treats the
/// unspecified addresses (`0.0.0.0` / `::`) and any concrete non-loopback host
/// as NOT loopback.
pub fn host_is_loopback(host: &str) -> bool {
    // Strip an IPv6 bracket form like "[::1]".
    let h = host.trim().trim_start_matches('[').trim_end_matches(']');
    if h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match h.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        // A hostname we can't resolve here — treat as non-loopback (be safe).
        Err(_) => false,
    }
}

/// Parse an ISO 8601 UTC timestamp into unix milliseconds. Accepts the exact
/// format emitted by `format_iso8601` (`YYYY-MM-DDTHH:MM:SSZ`), optionally
/// with fractional seconds (`.SSS`) and/or a numeric UTC offset (`+HH:MM` /
/// `-HH:MM`). Hand-rolled (no chrono); valid for 1970–~3000, like the
/// formatter. Used to honor the body `expires_at` retention convention
/// (#868) without pulling in a date library.
pub fn parse_iso8601_ms(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
    {
        return None;
    }
    let num = |r: std::ops::Range<usize>| -> Option<i64> {
        let t = s.get(r)?;
        if !t.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        t.parse().ok()
    };
    let year = num(0..4)?;
    let month = num(5..7)?;
    let day = num(8..10)?;
    let hour = num(11..13)?;
    let minute = num(14..16)?;
    let second = num(17..19)?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    // Optional fractional seconds.
    let mut frac_ms: i64 = 0;
    let mut i = 19;
    if b.get(i) == Some(&b'.') {
        i += 1;
        let start = i;
        while b.get(i).is_some_and(|c| c.is_ascii_digit()) {
            i += 1;
        }
        let digits = s.get(start..i)?;
        let mut v: i64 = digits.parse().ok()?;
        for _ in digits.len()..3 {
            v *= 10;
        }
        frac_ms = v;
    }
    // Optional numeric UTC offset (or 'Z').
    let mut offset_minutes: i64 = 0;
    match b.get(i) {
        Some(&b'Z') => {}
        Some(&b'+') | Some(&b'-') => {
            let sign: i64 = if b[i] == b'-' { -1 } else { 1 };
            let oh = num((i + 1)..(i + 3))?;
            let om = num((i + 4)..(i + 6))?;
            if oh > 23 || om > 59 {
                return None;
            }
            offset_minutes = sign * (oh * 60 + om);
        }
        _ => return None,
    }
    // days-from-civil (Hinnant): civil date -> days since 1970-01-01.
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    let secs = days * 86400 + hour * 3600 + minute * 60 + second - offset_minutes * 60;
    Some(secs * 1000 + frac_ms)
}

/// Format a unix timestamp in seconds as an ISO 8601 UTC string.
/// Avoids chrono dependency by hand-rolling a minimal formatter.
/// Only safe for timestamps from 1970 to ~3000 (no leap-second handling).
pub fn format_iso8601(secs: i64) -> String {
    if secs <= 0 {
        return "1970-01-01T00:00:00Z".to_string();
    }
    let days_since_epoch = secs / 86400;
    let secs_of_day = secs % 86400;
    let mut y = 1970i64;
    let mut d = days_since_epoch;
    loop {
        let days_in_year = if (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0) {
            366
        } else {
            365
        };
        if d < days_in_year {
            break;
        }
        d -= days_in_year;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 0usize;
    while m < 12 && d >= month_days[m] {
        d -= month_days[m];
        m += 1;
    }
    let month = m + 1;
    let day = d + 1;
    let h = secs_of_day / 3600;
    let min = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, month, day, h, min, s
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_time_eq_matches_semantics_of_plain_eq() {
        assert!(constant_time_str_eq("s3cret", "s3cret"));
        assert!(!constant_time_str_eq("s3cret", "s3creX"));
        assert!(!constant_time_str_eq("s3cret", "s3cret-longer"));
        assert!(!constant_time_str_eq("", "x"));
        assert!(constant_time_str_eq("", ""));
    }

    #[test]
    fn loopback_detection() {
        assert!(host_is_loopback("127.0.0.1"));
        assert!(host_is_loopback("127.5.6.7"));
        assert!(host_is_loopback("::1"));
        assert!(host_is_loopback("[::1]"));
        assert!(host_is_loopback("localhost"));
        assert!(!host_is_loopback("0.0.0.0"));
        assert!(!host_is_loopback("::"));
        assert!(!host_is_loopback("192.168.1.10"));
        assert!(!host_is_loopback("example.com"));
    }

    #[test]
    fn iso8601_roundtrip_and_offsets() {
        // #868: the expires_at body convention accepts ISO 8601 UTC.
        // Round-trip with the formatter (seconds precision).
        let fixed = 1_752_000_000i64;
        assert_eq!(parse_iso8601_ms(&format_iso8601(fixed)), Some(fixed * 1000));
        // Fractional seconds.
        let base = parse_iso8601_ms("2026-08-09T12:00:00Z").unwrap();
        assert_eq!(
            parse_iso8601_ms("2026-08-09T12:00:00.500Z"),
            Some(base + 500)
        );
        // Numeric UTC offsets normalize to the same instant.
        assert_eq!(parse_iso8601_ms("2026-08-09T08:00:00-04:00"), Some(base));
        assert_eq!(parse_iso8601_ms("2026-08-09T13:00:00+01:00"), Some(base));
        // Garbage and out-of-range values are rejected.
        assert_eq!(parse_iso8601_ms("not a date"), None);
        assert_eq!(parse_iso8601_ms("2026-13-09T12:00:00Z"), None);
        assert_eq!(parse_iso8601_ms("2026-08-09T25:00:00Z"), None);
        assert_eq!(parse_iso8601_ms("2026-08-09"), None);
    }
}
