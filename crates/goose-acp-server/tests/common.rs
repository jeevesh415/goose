use assert_json_diff::{assert_json_matches_no_panic, CompareMode, Config};
use goose::config::GooseMode;
use goose::model::ModelConfig;
use goose::providers::api_client::{ApiClient, AuthMethod};
use goose::providers::openai::OpenAiProvider;
use goose_acp_server::server::{serve, AcpServerConfig, GooseAcpAgent};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use rmcp::{
    handler::server::router::tool::ToolRouter, model::*, tool, tool_handler, tool_router,
    ErrorData as McpError, ServerHandler,
};
use sacp::schema::{McpServer, McpServerHttp, PermissionOptionKind, ToolCallStatus};
use std::collections::VecDeque;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

pub const FAKE_CODE: &str = "test-uuid-12345-67890";

/// Mock OpenAI streaming endpoint. Exchanges are (pattern, response) pairs.
/// On mismatch, returns 417 of the diff in OpenAI error format.
pub async fn setup_mock_openai(exchanges: Vec<(String, &'static str)>) -> MockServer {
    let mock_server = MockServer::start().await;
    let queue: VecDeque<(String, &'static str)> = exchanges.into_iter().collect();
    let queue = Arc::new(Mutex::new(queue));

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with({
            let queue = queue.clone();
            move |req: &wiremock::Request| {
                let body = String::from_utf8_lossy(&req.body);

                // Special case session rename request which doesn't happen in a predictable order.
                if body.contains("Reply with only a description in four words or less") {
                    return ResponseTemplate::new(200)
                        .insert_header("content-type", "application/json")
                        .set_body_string(include_str!(
                            "./test_data/openai_session_description.json"
                        ));
                }

                let (expected, response) = {
                    let mut q = queue.lock().unwrap();
                    q.pop_front().unwrap_or_default()
                };

                if body.contains(&expected) && !expected.is_empty() {
                    return ResponseTemplate::new(200)
                        .insert_header("content-type", "text/event-stream")
                        .set_body_string(response);
                }

                // Coerce non-json to allow a uniform JSON diff error response.
                let exp = serde_json::from_str(&expected)
                    .unwrap_or(serde_json::Value::String(expected.clone()));
                let act = serde_json::from_str(&body)
                    .unwrap_or(serde_json::Value::String(body.to_string()));
                let diff =
                    assert_json_matches_no_panic(&exp, &act, Config::new(CompareMode::Strict))
                        .unwrap_err();
                ResponseTemplate::new(417)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_json(serde_json::json!({"error": {"message": diff}}))
            }
        })
        .mount(&mock_server)
        .await;

    mock_server
}

#[derive(Clone)]
pub struct Lookup {
    tool_router: ToolRouter<Lookup>,
}

impl Default for Lookup {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_router]
impl Lookup {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    #[tool(description = "Get the code")]
    fn get_code(&self) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(FAKE_CODE)]))
    }
}

#[tool_handler]
impl ServerHandler for Lookup {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_03_26,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "lookup".into(),
                version: "1.0.0".into(),
                ..Default::default()
            },
            instructions: Some("Lookup server with get_code tool.".into()),
        }
    }
}

pub async fn spawn_mcp_http_server() -> (String, JoinHandle<()>) {
    let service = StreamableHttpService::new(
        || Ok(Lookup::new()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let url = format!("http://{addr}/mcp");

    let handle = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    (url, handle)
}

pub async fn spawn_acp_server_in_process(
    mock_server: &MockServer,
    builtins: &[&str],
    data_root: &Path,
    goose_mode: GooseMode,
) -> (
    tokio::io::DuplexStream,
    tokio::io::DuplexStream,
    JoinHandle<()>,
) {
    let api_client = ApiClient::new(
        mock_server.uri(),
        AuthMethod::BearerToken("test-key".to_string()),
    )
    .unwrap();
    let model_config = ModelConfig::new("gpt-5-nano").unwrap();
    let provider = OpenAiProvider::new(api_client, model_config);

    let config = AcpServerConfig {
        provider: Arc::new(provider),
        builtins: builtins.iter().map(|s| s.to_string()).collect(),
        work_dir: data_root.to_path_buf(),
        data_dir: data_root.to_path_buf(),
        config_dir: data_root.to_path_buf(),
        goose_mode,
    };

    let (client_read, server_write) = tokio::io::duplex(64 * 1024);
    let (server_read, client_write) = tokio::io::duplex(64 * 1024);

    let agent = Arc::new(GooseAcpAgent::with_config(config).await.unwrap());
    let handle = tokio::spawn(async move {
        if let Err(e) = serve(agent, server_read.compat(), server_write.compat_write()).await {
            tracing::error!("ACP server error: {e}");
        }
    });

    (client_read, client_write, handle)
}

pub async fn run_basic_completion_test<F, Fut>(run_prompt: F)
where
    F: FnOnce(Arc<MockServer>, PathBuf, GooseMode, String) -> Fut,
    Fut: Future<Output = String>,
{
    let temp_dir = tempfile::tempdir().unwrap();
    let prompt = "what is 1+1".to_string();
    let mock_server = setup_mock_openai(vec![(
        format!(r#"</info-msg>\n{prompt}""#),
        include_str!("./test_data/openai_basic_response.txt"),
    )])
    .await;

    let text = run_prompt(
        Arc::new(mock_server),
        temp_dir.path().to_path_buf(),
        GooseMode::Auto,
        prompt,
    )
    .await;
    assert!(text.contains("2"));
}

pub async fn run_mcp_http_server_test<F, Fut>(run_prompt: F)
where
    F: FnOnce(Arc<MockServer>, PathBuf, GooseMode, String, Vec<McpServer>) -> Fut,
    Fut: Future<Output = String>,
{
    let temp_dir = tempfile::tempdir().unwrap();
    let prompt = "Use the get_code tool and output only its result.".to_string();
    let (mcp_url, _handle) = spawn_mcp_http_server().await;

    let mock_server = setup_mock_openai(vec![
        (
            format!(r#"</info-msg>\n{prompt}""#),
            include_str!("./test_data/openai_tool_call_response.txt"),
        ),
        (
            format!(r#""content":"{FAKE_CODE}""#),
            include_str!("./test_data/openai_tool_result_response.txt"),
        ),
    ])
    .await;

    let text = run_prompt(
        Arc::new(mock_server),
        temp_dir.path().to_path_buf(),
        GooseMode::Auto,
        prompt,
        vec![McpServer::Http(McpServerHttp::new("lookup", mcp_url))],
    )
    .await;
    assert!(text.contains(FAKE_CODE));
}

pub async fn run_builtin_and_mcp_test<F, Fut>(run_prompt: F)
where
    F: FnOnce(
        Arc<MockServer>,
        PathBuf,
        GooseMode,
        String,
        Vec<McpServer>,
        Vec<&'static str>,
    ) -> Fut,
    Fut: Future<Output = ()>,
{
    let temp_dir = tempfile::tempdir().unwrap();
    let prompt =
        "Search for get_code and text_editor tools. Use them to save the code to /tmp/result.txt."
            .to_string();
    let (lookup_url, _lookup_handle) = spawn_mcp_http_server().await;

    let mock_server = setup_mock_openai(vec![
        (
            format!(r#"</info-msg>\n{prompt}""#),
            include_str!("./test_data/openai_builtin_search.txt"),
        ),
        (
            r#"lookup/get_code: Get the code"#.into(),
            include_str!("./test_data/openai_builtin_read_modules.txt"),
        ),
        (
            r#"lookup[\"get_code\"]({}): string - Get the code"#.into(),
            include_str!("./test_data/openai_builtin_execute.txt"),
        ),
        (
            r#"Successfully wrote to /tmp/result.txt"#.into(),
            include_str!("./test_data/openai_builtin_final.txt"),
        ),
    ])
    .await;

    let _ = fs_err::remove_file("/tmp/result.txt");
    run_prompt(
        Arc::new(mock_server),
        temp_dir.path().to_path_buf(),
        GooseMode::Auto,
        prompt,
        vec![McpServer::Http(McpServerHttp::new("lookup", lookup_url))],
        vec!["code_execution", "developer"],
    )
    .await;

    let result = fs_err::read_to_string("/tmp/result.txt").unwrap_or_default();
    assert!(result.contains(FAKE_CODE));
}

pub async fn run_permission_persistence_test<F, Fut>(run_prompt: F)
where
    F: Fn(
        Arc<MockServer>,
        PathBuf,
        GooseMode,
        String,
        Vec<McpServer>,
        Option<PermissionOptionKind>,
    ) -> Fut,
    Fut: Future<Output = ToolCallStatus>,
{
    let cases =
        vec![
        (
            Some(PermissionOptionKind::AllowAlways),
            ToolCallStatus::Completed,
            "user:\n  always_allow:\n  - lookup__get_code\n  ask_before: []\n  never_allow: []\n",
        ),
        (Some(PermissionOptionKind::AllowOnce), ToolCallStatus::Completed, ""),
        (
            Some(PermissionOptionKind::RejectAlways),
            ToolCallStatus::Failed,
            "user:\n  always_allow: []\n  ask_before: []\n  never_allow:\n  - lookup__get_code\n",
        ),
        (Some(PermissionOptionKind::RejectOnce), ToolCallStatus::Failed, ""),
        (None, ToolCallStatus::Failed, ""),
    ];

    for (kind, expected_status, expected_yaml) in cases {
        let temp_dir = tempfile::tempdir().unwrap();
        let prompt = "Use the get_code tool and output only its result.".to_string();
        let (mcp_url, _handle) = spawn_mcp_http_server().await;

        let mock_server = setup_mock_openai(vec![
            (
                format!(r#"</info-msg>\n{prompt}""#),
                include_str!("./test_data/openai_tool_call_response.txt"),
            ),
            (
                format!(r#""content":"{FAKE_CODE}""#),
                include_str!("./test_data/openai_tool_result_response.txt"),
            ),
        ])
        .await;

        let status = run_prompt(
            Arc::new(mock_server),
            temp_dir.path().to_path_buf(),
            GooseMode::Approve,
            prompt,
            vec![McpServer::Http(McpServerHttp::new("lookup", mcp_url))],
            kind,
        )
        .await;

        assert_eq!(status, expected_status);
        assert_eq!(
            fs_err::read_to_string(temp_dir.path().join("permission.yaml")).unwrap_or_default(),
            expected_yaml
        );
    }
}
