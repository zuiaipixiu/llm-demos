import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
  Tool,
} from "@modelcontextprotocol/sdk/types.js";

const CIP_URL = "https://cip.cc";

type CipIpInfo = {
  ip?: string;
  address?: string;
  operator?: string;
  isp?: string;
  data2?: string;
  data3?: string;
  url?: string;
};

const server = new Server(
  {
    name: "ipconfig-mcp",
    version: "0.1.0",
  },
  {
    capabilities: {
      resources: {},
      tools: {},
      logging: {},
    },
  }
);

const GET_LOCATION_IP_TOOL: Tool = {
  name: "get_location_ip",
  description: "Get current public IP address, geographic location, and ISP/operator information.",
  inputSchema: {
    type: "object",
    properties: {},
    additionalProperties: false,
  },
};

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [GET_LOCATION_IP_TOOL],
}));

function decodeHtmlEntities(value: string) {
  return value
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'");
}

function extractPreText(html: string) {
  const match = html.match(/<pre[^>]*>([\s\S]*?)<\/pre>/i);

  if (!match) {
    throw new Error("未找到 cip.cc 返回内容中的 <pre> 数据块");
  }

  return decodeHtmlEntities(match[1]).trim();
}

function parseCipText(text: string): CipIpInfo {
  const result = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .reduce<CipIpInfo>((parsed, line) => {
      const match = line.match(/^([^:：]+)\s*[:：]\s*(.*)$/);

      if (!match) {
        return parsed;
      }

      const [, rawKey, rawValue] = match;
      const keyMap: Record<string, keyof CipIpInfo> = {
        IP: "ip",
        地址: "address",
        运营商: "operator",
        数据二: "data2",
        数据三: "data3",
        URL: "url",
      };
      const key = keyMap[rawKey.trim()];

      if (key) {
        parsed[key] = rawValue.trim();
      }

      return parsed;
    }, {});

  if (result.operator && !result.isp) {
    result.isp = result.operator;
  }

  return result;
}

async function fetchIpInfo(): Promise<CipIpInfo> {
  const response = await fetch(CIP_URL, {
    headers: {
      "User-Agent": "mcp-get-location-ip/1.0",
    },
  });

  if (!response.ok) {
    throw new Error(`请求 ${CIP_URL} 失败: ${response.status} ${response.statusText}`);
  }

  const html = await response.text();
  const text = extractPreText(html);

  return parseCipText(text);
}

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "get_location_ip") {
    const ipInfo = await fetchIpInfo();

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(ipInfo, null, 2),
        },
      ],
    };
  }

  throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${request.params.name}`);
});

server.onerror = (error: any) => {
  console.error(error);
};

process.on("SIGINT", async () => {
  await server.close();
  process.exit(0);
});

async function runServer() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("MCP Starter Server running on stdio");
}

runServer().catch((error) => {
  console.error("Fatal error running server:", error);
  process.exit(1);
});
