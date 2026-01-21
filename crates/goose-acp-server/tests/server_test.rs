mod common;

use common::{
    run_basic_completion_test, run_builtin_and_mcp_test, run_mcp_http_server_test,
    run_permission_persistence_test, spawn_acp_server_in_process, FAKE_CODE,
};
use goose::config::GooseMode;
use sacp::schema::{
    ContentBlock, ContentChunk, InitializeRequest, McpServer, NewSessionRequest,
    PermissionOptionKind, PromptRequest, ProtocolVersion, RequestPermissionOutcome,
    RequestPermissionRequest, RequestPermissionResponse, SelectedPermissionOutcome,
    SessionNotification, SessionUpdate, StopReason, TextContent, ToolCallStatus,
};
use sacp::{ClientToAgent, JrConnectionCx};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use wiremock::MockServer;

#[tokio::test]
async fn test_acp_basic_completion() {
    run_basic_completion_test(|mock_server, data_root, mode, prompt| async move {
        run_acp_session(
            mock_server.as_ref(),
            vec![],
            &[],
            data_root.as_path(),
            mode,
            None,
            |cx, session_id, updates| async move {
                let response = cx
                    .send_request(PromptRequest::new(
                        session_id,
                        vec![ContentBlock::Text(TextContent::new(prompt))],
                    ))
                    .block_task()
                    .await
                    .unwrap();

                assert_eq!(response.stop_reason, StopReason::EndTurn);
                wait_for(
                    &updates,
                    &SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                        TextContent::new("2"),
                    ))),
                )
                .await;

                collect_agent_text(&updates)
            },
        )
        .await
    })
    .await;
}

#[tokio::test]
async fn test_acp_with_mcp_http_server() {
    run_mcp_http_server_test(
        |mock_server, data_root, mode, prompt, mcp_servers| async move {
            run_acp_session(
                mock_server.as_ref(),
                mcp_servers,
                &[],
                data_root.as_path(),
                mode,
                None,
                |cx, session_id, updates| async move {
                    let response = cx
                        .send_request(PromptRequest::new(
                            session_id,
                            vec![ContentBlock::Text(TextContent::new(prompt))],
                        ))
                        .block_task()
                        .await
                        .unwrap();

                    assert_eq!(response.stop_reason, StopReason::EndTurn);
                    wait_for(
                        &updates,
                        &SessionUpdate::AgentMessageChunk(ContentChunk::new(ContentBlock::Text(
                            TextContent::new(FAKE_CODE),
                        ))),
                    )
                    .await;

                    collect_agent_text(&updates)
                },
            )
            .await
        },
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_acp_with_builtin_and_mcp() {
    run_builtin_and_mcp_test(
        |mock_server, data_root, mode, prompt, mcp_servers, builtins| async move {
            run_acp_session(
                mock_server.as_ref(),
                mcp_servers,
                &builtins,
                data_root.as_path(),
                mode,
                None,
                |cx, session_id, _updates| async move {
                    let response = cx
                        .send_request(PromptRequest::new(
                            session_id,
                            vec![ContentBlock::Text(TextContent::new(prompt))],
                        ))
                        .block_task()
                        .await
                        .unwrap();

                    assert_eq!(response.stop_reason, StopReason::EndTurn);
                },
            )
            .await
        },
    )
    .await;
}

async fn wait_for(updates: &Arc<Mutex<Vec<SessionNotification>>>, expected: &SessionUpdate) {
    let deadline = tokio::time::Instant::now() + Duration::from_millis(500);
    let mut context = String::new();

    loop {
        let matched = {
            let guard = updates.lock().unwrap();
            context.clear();

            match expected {
                SessionUpdate::AgentMessageChunk(chunk) => {
                    let expected_text = match &chunk.content {
                        ContentBlock::Text(t) => &t.text,
                        other => panic!("wait_for: unhandled content {:?}", other),
                    };
                    for n in guard.iter() {
                        if let SessionUpdate::AgentMessageChunk(c) = &n.update {
                            if let ContentBlock::Text(t) = &c.content {
                                if t.text.is_empty() {
                                    context.clear();
                                } else {
                                    context.push_str(&t.text);
                                }
                            }
                        }
                    }
                    context.contains(expected_text)
                }
                SessionUpdate::ToolCallUpdate(expected_update) => {
                    for n in guard.iter() {
                        if let SessionUpdate::ToolCallUpdate(u) = &n.update {
                            context.push_str(&format!("{:?}\n", u));
                            if u.fields.status == expected_update.fields.status {
                                return;
                            }
                        }
                    }
                    false
                }
                other => panic!("wait_for: unhandled update {:?}", other),
            }
        };

        if matched {
            return;
        }
        if tokio::time::Instant::now() > deadline {
            panic!("Timeout waiting for {:?}\n\n{}", expected, context);
        }
        tokio::task::yield_now().await;
    }
}

fn collect_agent_text(updates: &Arc<Mutex<Vec<SessionNotification>>>) -> String {
    let guard = updates.lock().unwrap();
    let mut text = String::new();

    for notification in guard.iter() {
        if let SessionUpdate::AgentMessageChunk(chunk) = &notification.update {
            if let ContentBlock::Text(t) = &chunk.content {
                text.push_str(&t.text);
            }
        }
    }

    text
}

async fn wait_for_tool_status(updates: &Arc<Mutex<Vec<SessionNotification>>>) -> ToolCallStatus {
    let deadline = tokio::time::Instant::now() + Duration::from_millis(500);

    loop {
        if let Some(status) = {
            let guard = updates.lock().unwrap();
            guard.iter().find_map(|notification| {
                if let SessionUpdate::ToolCallUpdate(update) = &notification.update {
                    return update.fields.status;
                }
                None
            })
        } {
            return status;
        }

        if tokio::time::Instant::now() > deadline {
            panic!("Timeout waiting for ToolCallStatus");
        }
        tokio::task::yield_now().await;
    }
}

async fn run_acp_session<F, Fut, T>(
    mock_server: &MockServer,
    mcp_servers: Vec<McpServer>,
    builtins: &[&str],
    data_root: &Path,
    mode: GooseMode,
    select: Option<PermissionOptionKind>,
    test_fn: F,
) -> T
where
    F: FnOnce(
        JrConnectionCx<ClientToAgent>,
        sacp::schema::SessionId,
        Arc<Mutex<Vec<SessionNotification>>>,
    ) -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let (client_read, client_write, _handle) =
        spawn_acp_server_in_process(mock_server, builtins, data_root, mode).await;
    let work_dir = tempfile::tempdir().unwrap();
    let updates = Arc::new(Mutex::new(Vec::new()));
    let output: Arc<Mutex<Option<T>>> = Arc::new(Mutex::new(None));

    let transport = sacp::ByteStreams::new(client_write.compat_write(), client_read.compat());

    ClientToAgent::builder()
        .on_receive_notification(
            {
                let updates = updates.clone();
                async move |notification: SessionNotification, _cx| {
                    updates.lock().unwrap().push(notification);
                    Ok(())
                }
            },
            sacp::on_receive_notification!(),
        )
        .on_receive_request(
            async move |req: RequestPermissionRequest, request_cx, _connection_cx| {
                let response = match select {
                    Some(kind) => {
                        let id = req
                            .options
                            .iter()
                            .find(|o| o.kind == kind)
                            .unwrap()
                            .option_id
                            .clone();
                        RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
                            SelectedPermissionOutcome::new(id),
                        ))
                    }
                    None => RequestPermissionResponse::new(RequestPermissionOutcome::Cancelled),
                };
                request_cx.respond(response)
            },
            sacp::on_receive_request!(),
        )
        .connect_to(transport)
        .unwrap()
        .run_until({
            let updates = updates.clone();
            let output = output.clone();
            move |cx: JrConnectionCx<ClientToAgent>| async move {
                cx.send_request(InitializeRequest::new(ProtocolVersion::LATEST))
                    .block_task()
                    .await
                    .unwrap();

                let session = cx
                    .send_request(NewSessionRequest::new(work_dir.path()).mcp_servers(mcp_servers))
                    .block_task()
                    .await
                    .unwrap();

                let result = test_fn(cx.clone(), session.session_id, updates).await;
                *output.lock().unwrap() = Some(result);
                Ok(())
            }
        })
        .await
        .unwrap();

    let result = output.lock().unwrap().take().expect("missing test output");
    result
}

#[tokio::test]
async fn test_permission_persistence() {
    run_permission_persistence_test(
        |mock_server, data_root, mode, prompt, mcp_servers, kind| async move {
            run_acp_session(
                mock_server.as_ref(),
                mcp_servers,
                &[],
                data_root.as_path(),
                mode,
                kind,
                |cx, session_id, updates| async move {
                    cx.send_request(PromptRequest::new(
                        session_id,
                        vec![ContentBlock::Text(TextContent::new(prompt))],
                    ))
                    .block_task()
                    .await
                    .unwrap();
                    wait_for_tool_status(&updates).await
                },
            )
            .await
        },
    )
    .await;
}
