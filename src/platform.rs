//! Which operating systems a code path or capability applies to, and whether
//! it has been run there.
//!
//! What exists on one OS only sits behind `cfg(target_os)`. What exists on
//! several says per OS with a [`Support`] whether it was run there against
//! real drivers ([`Status::Tested`]) or only compiled and unit-tested
//! ([`Status::Untested`]).

use std::fmt;

/// An operating system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Os {
    Linux,
    Windows,
    /// macOS, the BSDs, ...: nothing is tested there.
    Other,
}

impl Os {
    /// The OS this was built for.
    pub const HOST: Os = if cfg!(target_os = "linux") {
        Os::Linux
    } else if cfg!(windows) {
        Os::Windows
    } else {
        Os::Other
    };

    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Os::Linux => "Linux",
            Os::Windows => "Windows",
            Os::Other => "other OS",
        }
    }
}

impl fmt::Display for Os {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How far something is known to work on one OS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Status {
    /// Run there against real hardware and drivers.
    Tested,
    /// Exists there and is unit-tested, but was never run against real
    /// drivers there.
    Untested,
    /// Does not exist there.
    Unavailable,
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Status::Tested => "tested",
            Status::Untested => "untested",
            Status::Unavailable => "unavailable",
        })
    }
}

/// [`Status`] on Linux and on Windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Support {
    pub linux: Status,
    pub windows: Status,
}

impl Support {
    /// Everywhere, run nowhere yet.
    pub const UNTESTED: Self = Self::new(Status::Untested, Status::Untested);
    /// Linux only, never run.
    pub const LINUX: Self = Self::new(Status::Untested, Status::Unavailable);
    /// Windows only, never run.
    pub const WINDOWS: Self = Self::new(Status::Unavailable, Status::Untested);
    /// Everywhere, run on Windows.
    pub const WINDOWS_TESTED: Self = Self::new(Status::Untested, Status::Tested);
    /// Windows only, run there.
    pub const WINDOWS_ONLY_TESTED: Self = Self::new(Status::Unavailable, Status::Tested);

    #[must_use]
    pub const fn new(linux: Status, windows: Status) -> Self {
        Self { linux, windows }
    }

    /// The status on `os`. Another OS has what both Linux and Windows have,
    /// untested; an OS-specific path is unavailable there.
    #[must_use]
    pub const fn on(self, os: Os) -> Status {
        match os {
            Os::Linux => self.linux,
            Os::Windows => self.windows,
            Os::Other => match (self.linux, self.windows) {
                (Status::Unavailable, _) | (_, Status::Unavailable) => Status::Unavailable,
                _ => Status::Untested,
            },
        }
    }

    /// The status on the OS this was built for.
    #[must_use]
    pub const fn here(self) -> Status {
        self.on(Os::HOST)
    }

    #[must_use]
    pub const fn available_on(self, os: Os) -> bool {
        !matches!(self.on(os), Status::Unavailable)
    }
}

impl fmt::Display for Support {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linux: {}, Windows: {}", self.linux, self.windows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn other_systems_get_what_both_have() {
        assert_eq!(Support::UNTESTED.on(Os::Other), Status::Untested);
        assert_eq!(Support::WINDOWS_TESTED.on(Os::Other), Status::Untested);
        assert_eq!(Support::LINUX.on(Os::Other), Status::Unavailable);
        assert_eq!(Support::WINDOWS.on(Os::Linux), Status::Unavailable);
        assert!(Support::LINUX.available_on(Os::Linux));
        assert_eq!(
            Support::WINDOWS_TESTED.to_string(),
            "Linux: untested, Windows: tested"
        );
    }
}
