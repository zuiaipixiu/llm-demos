const CIP_URL = "https://cip.cc";

function decodeHtmlEntities(value) {
  return value
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'"); 
}

function extractPreText(html) {
  const match = html.match(/<pre[^>]*>([\s\S]*?)<\/pre>/i);

  if (!match) {
    throw new Error("未找到 cip.cc 返回内容中的 <pre> 数据块");
  }

  return decodeHtmlEntities(match[1]).trim();
}

function parseCipText(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .reduce((result, line) => {
      const match = line.match(/^([^:：]+)\s*[:：]\s*(.*)$/);

      if (!match) {
        return result;
      }

      const [, rawKey, rawValue] = match;
      const keyMap = {
        IP: "ip",
        地址: "address",
        运营商: "operator",
        数据二: "data2",
        数据三: "data3",
        URL: "url",
      };
      const key = keyMap[rawKey.trim()] || rawKey.trim();

      result[key] = rawValue.trim();
      return result;
    }, {});
}

async function getCurrentIpInfo() {
  const response = await fetch(CIP_URL, {
    headers: {
      "User-Agent": "node-fetch-cip-parser/1.0",
    },
  });

  if (!response.ok) {
    throw new Error(`请求 ${CIP_URL} 失败: ${response.status} ${response.statusText}`);
  }

  const html = await response.text();
  const text = extractPreText(html);

  return parseCipText(text);
}

getCurrentIpInfo()
  .then((data) => {
    console.log(JSON.stringify(data, null, 2));
  })
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });

export { extractPreText, getCurrentIpInfo, parseCipText };
