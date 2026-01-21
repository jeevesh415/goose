use anyhow::Result;
use async_trait::async_trait;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

use crate::config::base::CodexCommand;
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

pub const CODEX_ACP_DEFAULT_MODEL: &str = "default";
pub const CODEX_ACP_DOC_URL: &str = "https://developers.openai.com/codex/cli";

#[derive(Debug)]
pub struct CodexAcpProvider {
    core: AcpProviderCore,
}

impl CodexAcpProvider {
    pub async fn from_env(model: ModelConfig) -> Result<Self> {
        let config = Config::global();
        let command: OsString = config.get_codex_command().unwrap_or_default().into();
        let resolved_command = SearchPaths::builder().with_npm().resolve(command)?;
        let goose_mode = config.get_goose_mode().unwrap_or(GooseMode::Auto);

        let permission_mapping = PermissionMapping {
            allow_option_id: Some("approved".to_string()),
            reject_option_id: Some("abort".to_string()),
            rejected_tool_status: ToolCallStatus::Failed,
        };

        let client_config = AcpClientConfig {
            command: resolved_command,
            args: vec![],
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
        GooseMode::Auto => "auto".to_string(),
        GooseMode::Approve => {
            // Best-fit: read-only requires approval for edits/commands, closest to manual mode.
            "read-only".to_string()
        }
        GooseMode::SmartApprove => {
            // Codex has no risk-based mode; read-only is the safest approximation.
            "read-only".to_string()
        }
        GooseMode::Chat => {
            // Codex lacks a no-tools mode; read-only is the closest available behavior.
            "read-only".to_string()
        }
    }
}

#[async_trait]
impl Provider for CodexAcpProvider {
    fn metadata() -> ProviderMetadata
    where
        Self: Sized,
    {
        ProviderMetadata::new(
            "codex-acp",
            "Codex ACP",
            "Use the Codex ACP agent over ACP.",
            CODEX_ACP_DEFAULT_MODEL,
            vec![],
            CODEX_ACP_DOC_URL,
            vec![ConfigKey::from_value_type::<CodexCommand>(true, false)],
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
