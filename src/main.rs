use anyhow::Result;
use llm_sdk::{Backend, Output};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Clone)]
struct LlmMcp {
    tool_router: ToolRouter<Self>,
}

impl LlmMcp {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CompleteParams {
    /// The prompt to send to the model
    prompt: String,

    /// Backend to use: "openrouter" or "openai" (default: "openrouter")
    backend: Option<String>,

    /// Model name (default: "google/gemini-2.5-flash")
    model: Option<String>,

    /// System prompt to prepend
    system_prompt: Option<String>,
}

fn build_backend(params: &CompleteParams) -> Result<Box<dyn Backend>> {
    let backend_name = params.backend.as_deref().unwrap_or("openrouter");
    let model = params
        .model
        .as_deref()
        .unwrap_or("google/gemini-2.5-flash");

    match backend_name {
        "openrouter" => {
            let mut b = llm_sdk::openrouter::OpenRouter::new(model)
                .api_key_env("OPENROUTER_API_KEY");
            if let Some(sp) = &params.system_prompt {
                b = b.system_prompt(sp.clone());
            }
            Ok(Box::new(b))
        }
        "openai" => {
            let mut b =
                llm_sdk::openai::OpenAI::new(model).api_key_env("OPENAI_API_KEY");
            if let Some(sp) = &params.system_prompt {
                b = b.system_prompt(sp.clone());
            }
            Ok(Box::new(b))
        }
        other => anyhow::bail!("unknown backend: {other}"),
    }
}

fn format_output(output: Output) -> String {
    let mut result = output.text;
    if let Some(usage) = &output.usage {
        result.push_str(&format!(
            "\n\n[tokens: {} in / {} out]",
            usage.input_tokens, usage.output_tokens
        ));
    }
    if let Some(cost) = output.cost_usd {
        result.push_str(&format!(" [${cost:.4}]"));
    }
    result
}

#[tool_router]
impl LlmMcp {
    #[tool(
        description = "Send a prompt to an LLM backend (OpenRouter, OpenAI). Use for delegating subtasks to cheaper/faster models."
    )]
    async fn complete(&self, Parameters(params): Parameters<CompleteParams>) -> String {
        match build_backend(&params) {
            Err(e) => format!("Error building backend: {e}"),
            Ok(backend) => match backend.complete(&params.prompt).await {
                Ok(output) => format_output(output),
                Err(e) => format!("Error: {e}"),
            },
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for LlmMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "LLM MCP server — delegate subtasks to cheaper models (OpenRouter, OpenAI). \
                 Call complete with a prompt and optional backend/model."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let service = LlmMcp::new();
    let server = service.serve(stdio()).await?;
    server.waiting().await?;
    Ok(())
}
