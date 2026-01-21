use anyhow::Result;
use async_trait::async_trait;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::base::ClaudeCodeAcpCommand;
use crate::config::search_path::SearchPaths;
use crate::config::{Config, GooseMode};
use crate::model::ModelConfig;
use crate::providers::acp_agent::AcpProviderCore;
use crate::providers::base::{
    ConfigKey, PermissionRouting, Provider, ProviderMetadata, ProviderUsage,
};
use crate::providers::errors::ProviderError;
use goose_acp_client::{schema::ToolCallStatus, AcpClient, AcpClientConfig, PermissionMapping};
use rmcp::model::Tool;

pub const CLAUDE_CODE_ACP_DEFAULT_MODEL: &str = "default";
pub const CLAUDE_CODE_ACP_DOC_URL: &str = "https://github.com/zed-industries/claude-code-acp";

#[derive(Debug)]
pub struct ClaudeCodeAcpProvider {
    core: AcpProviderCore,
}

impl ClaudeCodeAcpProvider {
    pub async fn from_env(model: ModelConfig) -> Result<Self> {
        let config = Config::global();
        let command: OsString = config
            .get_claude_code_acp_command()
            .unwrap_or_default()
            .into();
        let resolved_command = SearchPaths::builder().with_npm().resolve(command)?;
        let goose_mode = config.get_goose_mode().unwrap_or(GooseMode::Auto);

        let permission_mapping = PermissionMapping {
            allow_option_id: Some("allow".to_string()),
            reject_option_id: Some("reject".to_string()),
            rejected_tool_status: ToolCallStatus::Failed,
        };

        let client_config = AcpClientConfig {
            command: resolved_command,
            args: vec!["@zed-industries/claude-code-acp".to_string()],
            env: vec![],
            work_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            mcp_servers: vec![],
            session_mode_id: Some(map_goose_mode(goose_mode)),
            permission_mapping,
        };

        let client = AcpClient::connect(client_config).await?;
        Ok(Self {
            core: AcpProviderCore::with_client(
                Self::metadata().name,
                model,
                Arc::new(client),
                goose_mode,
            ),
        })
    }
}

fn map_goose_mode(goose_mode: GooseMode) -> String {
    match goose_mode {
        GooseMode::Auto => {
            // Closest to "autonomous": Claude Code's bypassPermissions skips confirmations.
            "bypassPermissions".to_string()
        }
        GooseMode::Approve => {
            // Claude Code's default matches "ask before risky actions".
            "default".to_string()
        }
        GooseMode::SmartApprove => {
            // Best-effort: acceptEdits auto-accepts file edits but still prompts for risky ops.
            "acceptEdits".to_string()
        }
        GooseMode::Chat => {
            // Plan mode disables tool execution, aligning with chat-only intent.
            "plan".to_string()
        }
    }
}

#[async_trait]
impl Provider for ClaudeCodeAcpProvider {
    fn metadata() -> ProviderMetadata
    where
        Self: Sized,
    {
        ProviderMetadata::new(
            "claude-code-acp",
            "Claude Code ACP",
            "Use the Claude Code ACP agent over ACP.",
            CLAUDE_CODE_ACP_DEFAULT_MODEL,
            vec![],
            CLAUDE_CODE_ACP_DOC_URL,
            vec![ConfigKey::from_value_type::<ClaudeCodeAcpCommand>(
                true, false,
            )],
        )
    }

    fn get_name(&self) -> &str {
        self.core.name()
    }

    fn get_model_config(&self) -> ModelConfig {
        self.core.model()
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn permission_routing(&self) -> PermissionRouting {
        PermissionRouting::ActionRequired
    }

    async fn handle_permission_confirmation(
        &self,
        request_id: &str,
        confirmation: &crate::permission::PermissionConfirmation,
    ) -> bool {
        self.core
            .handle_permission_confirmation(request_id, confirmation)
            .await
    }

    async fn complete_with_model(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[crate::conversation::message::Message],
        tools: &[Tool],
    ) -> Result<(crate::conversation::message::Message, ProviderUsage), ProviderError> {
        self.core
            .complete_with_model(model_config, system, messages, tools)
            .await
    }

    async fn stream(
        &self,
        system: &str,
        messages: &[crate::conversation::message::Message],
        tools: &[Tool],
    ) -> Result<crate::providers::base::MessageStream, ProviderError> {
        self.core.stream(system, messages, tools).await
    }
}
