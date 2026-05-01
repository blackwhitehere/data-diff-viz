use std::fs::OpenOptions;
use std::io::Write;

/// Initialize the logger to write to a file.
///
/// Logs are written to `app.log` in the current directory. Log level is
/// controlled by the `RUST_LOG` env var (default `info`).
pub fn init() -> Result<(), Box<dyn std::error::Error>> {
    init_with_path("app.log")
}

/// Initialize the logger with a custom log file path.
pub fn init_with_path(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let log_file = OpenOptions::new().create(true).append(true).open(path)?;

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| {
            use std::time::SystemTime;
            let secs = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            writeln!(
                buf,
                "[timestamp:{}] {} - {}",
                secs,
                record.level(),
                record.args()
            )
        })
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .try_init()?;

    Ok(())
}
