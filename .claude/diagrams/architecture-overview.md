# Architecture Overview

Top-level module dependency graph for `src/echo/`.

```mermaid
flowchart TD
    subgraph User["User / Host App"]
        APP[Application code]
    end

    subgraph Agents["agents"]
        BA[BaseAgent]
        GA[GenericAgent]
        SK[Skill]
    end

    subgraph LLM["llm"]
        LF[get_llm factory]
        BL[BaseLLM]
        PRV[Anthropic / OpenAI / Bedrock / Gemini]
    end

    subgraph Tools["tools"]
        BT[BaseTool]
        MCP[MCPConnectionManager + MCPTool]
        SKT[Skill meta-tools<br/>load_skill / unload_skill]
        PG[PgQueryTool]
    end

    subgraph Models["models"]
        CC[ConversationContext]
        MSG[Message / ToolCall / ToolResult]
    end

    subgraph Providers["External providers"]
        P1[Prompts: Langfuse]
        P2[Evals: Langfuse]
        P3[Audio: Gemini / EkaCare]
        P4[Postgres]
    end

    APP -->|build & run| GA
    GA -->|extends| BA
    BA -->|owns registry| SK
    BA -->|get_llm| LF --> BL --> PRV
    BA -->|tools per turn| BT
    BT --> MCP
    BT --> SKT
    BT --> PG
    BA -->|reads/returns| CC
    CC --> MSG
    PG --> P4
    APP -.optional.-> P1
    APP -.optional.-> P2
    APP -.optional.-> P3
```

**Read order for newcomers**: `models/` → `tools/base_tool.py` → `llm/base.py` → `agents/base.py` → one provider in `llm/` to see the agentic loop.
