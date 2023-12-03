fn main() -> Result<(), std::io::Error> {
    // TODO: FIXME: Remove all usage of `"/tmp"` in strings from the codebase.
    let tmp = std::path::Path::new("/tmp");
    if !tmp.is_dir() {
        std::fs::create_dir(tmp)?;
    };
    Ok(())
}
