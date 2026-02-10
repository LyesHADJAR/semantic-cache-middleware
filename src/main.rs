mod config;
mod error;
mod handlers;
mod models;
mod routes;
mod services;

use config::AppConfig;
use services::OllamaService;

#[tokio::main]
async fn main() {
    let config = AppConfig::from_env();
    let ollama = OllamaService::new(&config);
    let app = routes::app(ollama);

    let listener = tokio::net::TcpListener::bind(&config.listen_addr)
        .await
        .expect("failed to bind address");

    println!("Server running at http://{}", config.listen_addr);
    axum::serve(listener, app)
        .await
        .expect("failed to start server");
}
