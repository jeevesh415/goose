use anyhow::{Context, Result};
use sacp::schema::{
    ContentBlock, ContentChunk, InitializeRequest, McpServer, NewSessionRequest, PromptRequest,
    ProtocolVersion, RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SessionNotification, SessionUpdate, SetSessionModeRequest, StopReason, TextContent,
    ToolCallContent, ToolCallStatus,
};
use sacp::{ClientToAgent, JrConnectionCx};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex as TokioMutex};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::provider::{map_permission_response, PermissionDecision, PermissionMapping};
#[derive(Clone, Debug)]
pub struct AcpClientConfig {
    pub command: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub work_dir: PathBuf,
    pub mcp_servers: Vec<McpServer>,
    pub session_mode_id: Option<String>,
    pub permission_mapping: PermissionMapping,
}

impl Default for AcpClientConfig {
    fn default() -> Self {
        Self {
            command: PathBuf::from("goose"),
            args: vec!["acp".to_string()],
            env: vec![],
            work_dir: std::env::current_dir().unwrap_or_default(),
            mcp_servers: vec![],
            session_mode_id: None,
            permission_mapping: PermissionMapping::default(),
        }
    }
}

#[derive(Clone)]
pub struct AcpClient {
    tx: mpsc::Sender<ClientRequest>,
    permission_mapping: PermissionMapping,
    rejected_tool_calls: Arc<TokioMutex<HashSet<String>>>,
}

enum ClientRequest {
    Prompt {
        content: Vec<ContentBlock>,
        response_tx: mpsc::Sender<AcpUpdate>,
    },
    Shutdown,
}

#[derive(Debug)]
pub enum AcpUpdate {
    Text(String),
    Thought(String),
    ToolCallStart {
        id: String,
        title: String,
        raw_input: Option<serde_json::Value>,
    },
    ToolCallComplete {
        id: String,
        status: ToolCallStatus,
        content: Vec<ToolCallContent>,
    },
    PermissionRequest {
        request: Box<RequestPermissionRequest>,
        response_tx: oneshot::Sender<RequestPermissionResponse>,
    },
    Complete(StopReason),
    Error(String),
}

impl AcpClient {
    pub async fn connect(config: AcpClientConfig) -> Result<Self> {
        let (tx, rx) = mpsc::channel(32);
        let (init_tx, init_rx) = oneshot::channel();
        let permission_mapping = config.permission_mapping.clone();
        let rejected_tool_calls = Arc::new(TokioMutex::new(HashSet::new()));

        tokio::spawn(run_client_loop(config, rx, init_tx));

        init_rx
            .await
            .context("ACP client initialization cancelled")??;

        Ok(Self {
            tx,
            permission_mapping,
            rejected_tool_calls,
        })
    }

    /// Connect with an explicit transport (for in-process testing).
    pub async fn connect_with_transport<R, W>(
        config: AcpClientConfig,
        read: R,
        write: W,
    ) -> Result<Self>
    where
        R: futures::AsyncRead + Unpin + Send + 'static,
        W: futures::AsyncWrite + Unpin + Send + 'static,
    {
        let (tx, mut rx) = mpsc::channel(32);
        let (init_tx, init_rx) = oneshot::channel();
        let permission_mapping = config.permission_mapping.clone();
        let rejected_tool_calls = Arc::new(TokioMutex::new(HashSet::new()));
        let transport = sacp::ByteStreams::new(write, read);
        let init_tx = Arc::new(Mutex::new(Some(init_tx)));
        tokio::spawn(async move {
            if let Err(e) =
                run_protocol_loop_with_transport(config, transport, &mut rx, init_tx.clone()).await
            {
                tracing::error!("ACP protocol error: {e}");
            }
        });

        init_rx
            .await
            .context("ACP client initialization cancelled")??;

        Ok(Self {
            tx,
            permission_mapping,
            rejected_tool_calls,
        })
    }

    pub async fn prompt(&self, content: Vec<ContentBlock>) -> Result<mpsc::Receiver<AcpUpdate>> {
        let (response_tx, response_rx) = mpsc::channel(64);
        self.tx
            .send(ClientRequest::Prompt {
                content,
                response_tx,
            })
            .await
            .context("ACP client is unavailable")?;
        Ok(response_rx)
    }

    pub async fn permission_response(
        &self,
        request: &RequestPermissionRequest,
        decision: PermissionDecision,
    ) -> RequestPermissionResponse {
        if decision.should_record_rejection() {
            self.rejected_tool_calls
                .lock()
                .await
                .insert(request.tool_call.tool_call_id.0.to_string());
        }

        map_permission_response(&self.permission_mapping, request, decision)
    }

    pub async fn tool_call_is_error(&self, tool_call_id: &str, status: ToolCallStatus) -> bool {
        let was_rejected = self.rejected_tool_calls.lock().await.remove(tool_call_id);

        match status {
            ToolCallStatus::Failed => true,
            ToolCallStatus::Completed => {
                was_rejected
                    && self.permission_mapping.rejected_tool_status == ToolCallStatus::Completed
            }
            _ => false,
        }
    }
}

impl Drop for AcpClient {
    fn drop(&mut self) {
        let tx = self.tx.clone();
        tokio::spawn(async move {
            let _ = tx.send(ClientRequest::Shutdown).await;
        });
    }
}

async fn run_client_loop(
    config: AcpClientConfig,
    mut rx: mpsc::Receiver<ClientRequest>,
    init_tx: oneshot::Sender<Result<()>>,
) {
    let init_tx = Arc::new(Mutex::new(Some(init_tx)));

    let child = match spawn_acp_process(&config).await {
        Ok(c) => c,
        Err(e) => {
            let message = e.to_string();
            send_init_result(&init_tx, Err(anyhow::anyhow!(message.clone())));
            tracing::error!("failed to spawn ACP process: {message}");
            return;
        }
    };

    if let Err(e) = run_protocol_loop_with_child(config, child, &mut rx, init_tx.clone()).await {
        let message = e.to_string();
        send_init_result(&init_tx, Err(anyhow::anyhow!(message.clone())));
        tracing::error!("ACP protocol error: {message}");
    }
}

async fn spawn_acp_process(config: &AcpClientConfig) -> Result<Child> {
    let mut cmd = Command::new(&config.command);
    cmd.args(&config.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true);

    for (key, value) in &config.env {
        cmd.env(key, value);
    }

    cmd.spawn().context("failed to spawn ACP process")
}

async fn run_protocol_loop_with_child(
    config: AcpClientConfig,
    mut child: Child,
    rx: &mut mpsc::Receiver<ClientRequest>,
    init_tx: Arc<Mutex<Option<oneshot::Sender<Result<()>>>>>,
) -> Result<()> {
    let stdin = child.stdin.take().context("no stdin")?;
    let stdout = child.stdout.take().context("no stdout")?;
    let transport = sacp::ByteStreams::new(stdin.compat_write(), stdout.compat());
    run_protocol_loop_with_transport(config, transport, rx, init_tx).await
}

async fn run_protocol_loop_with_transport<R, W>(
    config: AcpClientConfig,
    transport: sacp::ByteStreams<W, R>,
    rx: &mut mpsc::Receiver<ClientRequest>,
    init_tx: Arc<Mutex<Option<oneshot::Sender<Result<()>>>>>,
) -> Result<()>
where
    R: futures::AsyncRead + Unpin + Send + 'static,
    W: futures::AsyncWrite + Unpin + Send + 'static,
{
    let prompt_response_tx: Arc<Mutex<Option<mpsc::Sender<AcpUpdate>>>> =
        Arc::new(Mutex::new(None));

    ClientToAgent::builder()
        .on_receive_notification(
            {
                let prompt_response_tx = prompt_response_tx.clone();
                async move |notification: SessionNotification, _cx| {
                    if let Some(tx) = prompt_response_tx.lock().unwrap().as_ref() {
                        match notification.update {
                            SessionUpdate::AgentMessageChunk(ContentChunk {
                                content: ContentBlock::Text(TextContent { text, .. }),
                                ..
                            }) => {
                                let _ = tx.try_send(AcpUpdate::Text(text));
                            }
                            SessionUpdate::AgentThoughtChunk(ContentChunk {
                                content: ContentBlock::Text(TextContent { text, .. }),
                                ..
                            }) => {
                                let _ = tx.try_send(AcpUpdate::Thought(text));
                            }
                            SessionUpdate::ToolCall(tool_call) => {
                                let _ = tx.try_send(AcpUpdate::ToolCallStart {
                                    id: tool_call.tool_call_id.0.to_string(),
                                    title: tool_call.title,
                                    raw_input: tool_call.raw_input,
                                });
                            }
                            SessionUpdate::ToolCallUpdate(update) => {
                                if let Some(status) = update.fields.status {
                                    let _ = tx.try_send(AcpUpdate::ToolCallComplete {
                                        id: update.tool_call_id.0.to_string(),
                                        status,
                                        content: update.fields.content.unwrap_or_default(),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                    Ok(())
                }
            },
            sacp::on_receive_notification!(),
        )
        .on_receive_request(
            {
                let prompt_response_tx = prompt_response_tx.clone();
                async move |request: RequestPermissionRequest, request_cx, _connection_cx| {
                    let (response_tx, response_rx) = oneshot::channel();

                    let handler = prompt_response_tx.lock().unwrap().as_ref().cloned();
                    let tx = handler.ok_or_else(sacp::Error::internal_error)?;

                    if tx.is_closed() {
                        return Err(sacp::Error::internal_error());
                    }

                    tx.try_send(AcpUpdate::PermissionRequest {
                        request: Box::new(request),
                        response_tx,
                    })
                    .map_err(|_| sacp::Error::internal_error())?;

                    let response = response_rx.await.unwrap_or_else(|_| {
                        RequestPermissionResponse::new(RequestPermissionOutcome::Cancelled)
                    });
                    request_cx.respond(response)
                }
            },
            sacp::on_receive_request!(),
        )
        .connect_to(transport)?
        .run_until({
            let prompt_response_tx = prompt_response_tx.clone();
            move |cx: JrConnectionCx<ClientToAgent>| {
                handle_requests(config, cx, rx, prompt_response_tx, init_tx.clone())
            }
        })
        .await?;

    Ok(())
}

async fn handle_requests(
    config: AcpClientConfig,
    cx: JrConnectionCx<ClientToAgent>,
    rx: &mut mpsc::Receiver<ClientRequest>,
    prompt_response_tx: Arc<Mutex<Option<mpsc::Sender<AcpUpdate>>>>,
    init_tx: Arc<Mutex<Option<oneshot::Sender<Result<()>>>>>,
) -> Result<(), sacp::Error> {
    cx.send_request(InitializeRequest::new(ProtocolVersion::LATEST))
        .block_task()
        .await
        .map_err(|err| {
            let message = format!("ACP initialize failed: {err}");
            send_init_result(&init_tx, Err(anyhow::anyhow!(message.clone())));
            sacp::Error::internal_error().data(message)
        })?;

    let session = cx
        .send_request(NewSessionRequest::new(config.work_dir).mcp_servers(config.mcp_servers))
        .block_task()
        .await
        .map_err(|err| {
            let message = format!("ACP session/new failed: {err}");
            send_init_result(&init_tx, Err(anyhow::anyhow!(message.clone())));
            sacp::Error::internal_error().data(message)
        })?;

    let session_id = session.session_id;
    if let Some(mode_id) = config.session_mode_id {
        let modes = session.modes.ok_or_else(|| {
            let message = "ACP agent did not advertise SessionModeState".to_string();
            send_init_result(&init_tx, Err(anyhow::anyhow!(message.clone())));
            sacp::Error::invalid_params().data(message)
        })?;

        if modes.current_mode_id.0.as_ref() != mode_id.as_str() {
            let available: Vec<String> = modes
                .available_modes
                .iter()
                .map(|mode| mode.id.0.to_string())
                .collect();

            if !available.iter().any(|id| id == &mode_id) {
                let message = format!(
                    "Requested mode '{}' not offered by agent. Available modes: {}",
                    mode_id,
                    available.join(", ")
                );
                send_init_result(&init_tx, Err(anyhow::anyhow!(message.clone())));
                return Err(sacp::Error::invalid_params().data(message));
            }

            if let Err(err) = cx
                .send_request(SetSessionModeRequest::new(session_id.clone(), mode_id))
                .block_task()
                .await
            {
                let message = format!("ACP agent rejected session/set_mode: {err}");
                send_init_result(&init_tx, Err(anyhow::anyhow!(message.clone())));
                return Err(sacp::Error::internal_error().data(message));
            }
        }
    }

    send_init_result(&init_tx, Ok(()));

    while let Some(request) = rx.recv().await {
        match request {
            ClientRequest::Prompt {
                content,
                response_tx,
            } => {
                *prompt_response_tx.lock().unwrap() = Some(response_tx.clone());

                let response = cx
                    .send_request(PromptRequest::new(session_id.clone(), content))
                    .block_task()
                    .await;

                match response {
                    Ok(r) => {
                        let _ = response_tx.try_send(AcpUpdate::Complete(r.stop_reason));
                    }
                    Err(e) => {
                        let _ = response_tx.try_send(AcpUpdate::Error(e.to_string()));
                    }
                }

                *prompt_response_tx.lock().unwrap() = None;
            }
            ClientRequest::Shutdown => break,
        }
    }

    Ok(())
}

fn send_init_result(init_tx: &Arc<Mutex<Option<oneshot::Sender<Result<()>>>>>, result: Result<()>) {
    if let Some(tx) = init_tx.lock().unwrap().take() {
        let _ = tx.send(result);
    }
}

pub fn text_content(text: impl Into<String>) -> ContentBlock {
    ContentBlock::Text(TextContent::new(text))
}
