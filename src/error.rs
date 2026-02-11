use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Unified error type for the application.
#[derive(Debug)]
pub enum AppError {
    /// An upstream HTTP request failed.
    Request(reqwest::Error),
    /// Failed to deserialise / parse a response body.
    ResponseParse(String),
    /// Upstream returned a successful status but the body contained no useful content.
    EmptyResponse,
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Request(err) => write!(f, "Request error: {err}"),
            Self::ResponseParse(msg) => write!(f, "Response error: {msg}"),
            Self::EmptyResponse => write!(f, "Upstream returned an empty response"),
        }
    }
}

impl std::error::Error for AppError {}

/// Convert `reqwest::Error` into our `AppError` automatically via `?`.
impl From<reqwest::Error> for AppError {
    fn from(err: reqwest::Error) -> Self {
        Self::Request(err)
    }
}

/// Let Axum turn an `AppError` into an HTTP response directly.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = match &self {
            Self::Request(_) => StatusCode::BAD_GATEWAY,
            Self::ResponseParse(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::EmptyResponse => StatusCode::BAD_GATEWAY,
        };
        (status, self.to_string()).into_response()
    }
}
