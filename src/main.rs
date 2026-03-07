use anyhow::Result;
use llm_agent::{AgentLoop, HookContext, HookDecision, ToolHook};
use llm_sdk::tools::ToolSet;
use llm_sdk::{Output, TokenUsage};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// --- Config ---

#[derive(Debug, Deserialize)]
struct Config {
    #[serde(default = "default_backend")]
    backend: String,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default = "default_max_turns")]
    max_turns: u32,
}

fn default_backend() -> String { "codex".into() }
fn default_model() -> String { "gpt-5.4".into() }
fn default_max_turns() -> u32 { 20 }

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            model: default_model(),
            system_prompt: None,
            max_turns: default_max_turns(),
        }
    }
}

impl Config {
    fn load() -> Self {
        let path = config_path();
        match std::fs::read_to_string(&path) {
            Ok(contents) => toml::from_str(&contents).unwrap_or_else(|e| {
                eprintln!("llm-mcp: bad config {}: {e}", path.display());
                Config::default()
            }),
            Err(_) => Config::default(),
        }
    }
}

fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("llm-mcp")
        .join("config.toml")
}

// --- MCP server ---

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
    /// Backend override: "codex", "openrouter", or "openai"
    backend: Option<String>,
    /// Model name override
    model: Option<String>,
    /// System prompt override
    system_prompt: Option<String>,
}

// --- BashHook: shells out to claude-bash-hook ---

struct BashHook;

#[derive(Serialize)]
struct HookInput {
    tool_name: String,
    tool_input: HookToolInput,
}

#[derive(Serialize)]
struct HookToolInput {
    command: Option<String>,
}

#[derive(Deserialize)]
struct HookOutput {
    #[serde(rename = "hookSpecificOutput")]
    hook_output: HookSpecificOutput,
}

#[derive(Deserialize)]
struct HookSpecificOutput {
    #[serde(rename = "permissionDecision")]
    decision: String,
    #[serde(rename = "permissionDecisionReason")]
    reason: String,
}

impl BashHook {
    async fn call_hook(command: &str) -> Result<HookOutput> {
        use tokio::io::AsyncWriteExt;

        let input = HookInput {
            tool_name: "Bash".into(),
            tool_input: HookToolInput {
                command: Some(command.into()),
            },
        };
        let json = serde_json::to_string(&input)?;
        let mut child = tokio::process::Command::new("claude-bash-hook")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(json.as_bytes()).await?;
            stdin.shutdown().await?;
        }

        let output = child.wait_with_output().await?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(serde_json::from_str(&stdout)?)
    }
}

#[async_trait::async_trait]
impl ToolHook for BashHook {
    async fn pre_execute(
        &self,
        ctx: &HookContext<'_>,
    ) -> Result<HookDecision, Box<dyn std::error::Error + Send + Sync>> {
        if ctx.tool_name != "Bash" {
            return Ok(HookDecision::Allow);
        }
        let command = match extract_command(ctx.arguments) {
            Some(c) => c,
            None => return Ok(HookDecision::Allow),
        };
        match Self::call_hook(&command).await {
            Ok(output) => match output.hook_output.decision.as_str() {
                "allow" => Ok(HookDecision::Allow),
                "deny" | "block" => Ok(HookDecision::Block(output.hook_output.reason)),
                "ask" => Ok(HookDecision::Block(format!(
                    "Requires confirmation: {}",
                    output.hook_output.reason
                ))),
                _ => Ok(HookDecision::Allow),
            },
            Err(_) => Ok(HookDecision::Allow),
        }
    }
}

fn extract_command(arguments: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get("command").and_then(|c| c.as_str()).map(String::from))
}

// --- ToolSet executor adapter ---

struct ToolSetExecutor<'a> {
    tool_set: &'a ToolSet,
}

#[async_trait::async_trait]
impl llm_agent::ToolExecutor for ToolSetExecutor<'_> {
    async fn execute(&self, name: &str, arguments: &str) -> String {
        let call = llm_sdk::tools::ToolCall {
            id: String::new(),
            name: name.to_string(),
            arguments: arguments.to_string(),
        };
        self.tool_set.execute(&call).await
    }
}

// --- Completion logic ---

fn tools_json(tool_set: &ToolSet) -> serde_json::Value {
    let defs: Vec<serde_json::Value> = tool_set
        .definitions()
        .into_iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })
        })
        .collect();
    serde_json::Value::Array(defs)
}

fn to_output(output: llm_agent::AgentOutput) -> Output {
    Output {
        text: output.text(),
        usage: Some(TokenUsage {
            input_tokens: output.usage.input_tokens,
            output_tokens: output.usage.output_tokens,
            ..Default::default()
        }),
        session_id: None,
        cost_usd: None,
    }
}

async fn run_with_client<C: llm_agent::ChatClient>(
    client: &C,
    prompt: &str,
    system_prompt: Option<&str>,
    max_turns: u32,
) -> Result<Output> {
    let tool_set = ToolSet::standard();
    let tj = tools_json(&tool_set);
    let executor = ToolSetExecutor { tool_set: &tool_set };

    let mut agent = AgentLoop::new(client, executor)
        .with_hook(BashHook)
        .max_turns(max_turns)
        .tools_json(tj);

    if let Some(sp) = system_prompt {
        agent = agent.system_prompt(sp);
    }

    let output = agent
        .run(prompt)
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(to_output(output))
}

async fn run_completion(params: &CompleteParams) -> Result<Output> {
    let config = Config::load();
    let backend = params.backend.as_deref().unwrap_or(&config.backend);
    let model = params.model.as_deref().unwrap_or(&config.model);
    let system_prompt = params
        .system_prompt
        .as_deref()
        .or(config.system_prompt.as_deref());

    match backend {
        "openrouter" => {
            let client = llm_sdk::openrouter::OpenRouter::new(model)
                .api_key_env("OPENROUTER_API_KEY");
            run_with_client(&client, &params.prompt, system_prompt, config.max_turns).await
        }
        "openai" => {
            let client = llm_sdk::openai::OpenAI::new(model)
                .api_key_env("OPENAI_API_KEY");
            run_with_client(&client, &params.prompt, system_prompt, config.max_turns).await
        }
        "codex" => {
            let client = llm_sdk::codex::Codex::new(model);
            run_with_client(&client, &params.prompt, system_prompt, config.max_turns).await
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

// --- MCP tool ---

#[tool_router]
impl LlmMcp {
    #[tool(
        description = "Send a prompt to an LLM backend (Codex, OpenRouter, OpenAI). The model can use Bash, Read, Write, Glob, Grep tools. Bash commands are validated by claude-bash-hook. Defaults to Codex gpt-5.4."
    )]
    async fn complete(&self, Parameters(params): Parameters<CompleteParams>) -> String {
        match run_completion(&params).await {
            Ok(output) => format_output(output),
            Err(e) => format!("Error: {e}"),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for LlmMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "LLM MCP server — delegate subtasks to other models (Codex/ChatGPT Pro, OpenRouter, OpenAI). \
                 Call complete with a prompt. Defaults to Codex (gpt-5.4) using ChatGPT subscription. \
                 Config at ~/.config/llm-mcp/config.toml, overridable per-call."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 && args[1] != "mcp-server" {
        let prompt = args[1..].join(" ");
        return run_cli(&prompt).await;
    }

    let service = LlmMcp::new();
    let server = service.serve(stdio()).await?;
    server.waiting().await?;
    Ok(())
}

async fn run_cli(prompt: &str) -> Result<()> {
    let params = CompleteParams {
        prompt: prompt.to_string(),
        backend: None,
        model: None,
        system_prompt: None,
    };
    match run_completion(&params).await {
        Ok(output) => print!("{}", format_output(output)),
        Err(e) => eprintln!("Error: {e}"),
    }
    Ok(())
}
