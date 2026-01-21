mod common;

use common::{
    run_basic_completion_test, run_builtin_and_mcp_test, run_mcp_http_server_test,
    run_permission_persistence_test, spawn_acp_server_in_process,
};
use futures::StreamExt;
use goose::config::GooseMode;
use goose::conversation::message::{ActionRequiredData, Message, MessageContent};
use goose::model::ModelConfig;
use goose::permission::permission_confirmation::PrincipalType;
use goose::permission::{Permission, PermissionConfirmation};
use goose::providers::acp_agent::AcpProviderCore;
use goose_acp_client::{
    schema::{McpServer, PermissionOptionKind, ToolCallStatus},
    AcpClient, AcpClientConfig, PermissionMapping,
};
use std::path::Path;
use std::sync::Arc;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

async fn setup_provider(
    mock_server: &wiremock::MockServer,
    data_root: &Path,
    builtins: &[&str],
    mcp_servers: Vec<McpServer>,
    goose_mode: GooseMode,
) -> AcpProviderCore {
    let (client_read, client_write, _handle) =
        spawn_acp_server_in_process(mock_server, builtins, data_root, goose_mode).await;

    let client_config = AcpClientConfig {
        command: "unused".into(),
        args: vec![],
        env: vec![],
        work_dir: data_root.to_path_buf(),
        mcp_servers,
        session_mode_id: None,
        permission_mapping: PermissionMapping::default(),
    };

    let client = AcpClient::connect_with_transport(
        client_config,
        client_read.compat(),
        client_write.compat_write(),
    )
    .await
    .unwrap();

    AcpProviderCore::with_client(
        "acp-test".to_string(),
        ModelConfig::new("default").unwrap(),
        Arc::new(client),
        goose_mode,
    )
}

async fn stream_text(
    provider: &AcpProviderCore,
    prompt: &str,
    permission: Option<PermissionOptionKind>,
) -> (String, bool) {
    let message = Message::user().with_text(prompt);
    let mut stream = provider.stream("", &[message], &[]).await.unwrap();
    let mut text = String::new();
    let mut tool_error = false;

    while let Some(item) = stream.next().await {
        let (msg, _) = item.unwrap();
        if let Some(msg) = msg {
            for content in msg.content {
                match content {
                    MessageContent::Text(t) => {
                        text.push_str(&t.text);
                    }
                    MessageContent::ToolResponse(resp) => {
                        if let Ok(result) = resp.tool_result {
                            tool_error = result.is_error.unwrap_or(false);
                        }
                    }
                    MessageContent::ActionRequired(action) => {
                        if let ActionRequiredData::ToolConfirmation { id, .. } = action.data {
                            let permission = permission
                                .map(|kind| match kind {
                                    PermissionOptionKind::AllowAlways => Permission::AlwaysAllow,
                                    PermissionOptionKind::AllowOnce => Permission::AllowOnce,
                                    PermissionOptionKind::RejectAlways => Permission::AlwaysDeny,
                                    PermissionOptionKind::RejectOnce => Permission::DenyOnce,
                                    _ => Permission::Cancel,
                                })
                                .unwrap_or(Permission::Cancel);

                            let confirmation = PermissionConfirmation {
                                principal_type: PrincipalType::Tool,
                                permission,
                            };

                            let handled = provider
                                .handle_permission_confirmation(&id, &confirmation)
                                .await;
                            assert!(handled);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    (text, tool_error)
}

#[tokio::test]
async fn test_provider_basic_completion() {
    run_basic_completion_test(|mock_server, data_root, mode, prompt| async move {
        let provider =
            setup_provider(mock_server.as_ref(), data_root.as_path(), &[], vec![], mode).await;
        let (text, _tool_error) = stream_text(&provider, &prompt, None).await;
        text
    })
    .await;
}

#[tokio::test]
async fn test_provider_with_mcp_http_server() {
    run_mcp_http_server_test(
        |mock_server, data_root, mode, prompt, mcp_servers| async move {
            let provider = setup_provider(
                mock_server.as_ref(),
                data_root.as_path(),
                &[],
                mcp_servers,
                mode,
            )
            .await;
            let (text, _tool_error) = stream_text(&provider, &prompt, None).await;
            text
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_provider_with_builtin_and_mcp() {
    run_builtin_and_mcp_test(
        |mock_server, data_root, mode, prompt, mcp_servers, builtins| async move {
            let provider = setup_provider(
                mock_server.as_ref(),
                data_root.as_path(),
                &builtins,
                mcp_servers,
                mode,
            )
            .await;
            let _ = stream_text(&provider, &prompt, None).await;
        },
    )
    .await;
}

#[tokio::test]
async fn test_permission_persistence() {
    run_permission_persistence_test(
        |mock_server, data_root, mode, prompt, mcp_servers, kind| async move {
            let provider = setup_provider(
                mock_server.as_ref(),
                data_root.as_path(),
                &[],
                mcp_servers,
                mode,
            )
            .await;
            let (_text, tool_error) = stream_text(&provider, &prompt, kind).await;
            if tool_error {
                ToolCallStatus::Failed
            } else {
                ToolCallStatus::Completed
            }
        },
    )
    .await;
}
