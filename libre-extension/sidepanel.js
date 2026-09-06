const API = "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);
let shot = null; // data URL of the captured tab
// In-memory conversation for THIS session. Sent in full every turn so the model
// keeps context (and stays anchored on the session's screenshots). Not persisted —
// cleared on "New" or when the side panel closes.
let conversation = [];

// ---- daemon health indicator ----
async function ping() {
  try {
    const r = await fetch(`${API}/health`, { cache: "no-store" });
    const j = await r.json();
    $("status").className = "dot " + (j.status === "ok" ? "ok" : "bad");
  } catch {
    $("status").className = "dot bad";
  }
}
ping();
setInterval(ping, 5000);

// ---- capture the visible tab (delegated to the service worker) ----
$("capture").addEventListener("click", () => {
  chrome.runtime.sendMessage({ cmd: "capture" }, (res) => {
    if (!res || res.error) {
      addMsg("bot", "Capture failed: " + (res && res.error || "unknown"));
      return;
    }
    shot = res.dataUrl;
    $("thumb").src = shot;
    $("preview").classList.remove("hidden");
    $("capture").classList.add("has-shot");
    $("capture").textContent = "📸 Tab captured — ready";
  });
});

$("clearShot").addEventListener("click", clearShot);
function clearShot() {
  shot = null;
  $("preview").classList.add("hidden");
  $("capture").classList.remove("has-shot");
  $("capture").textContent = "📸 Capture tab";
}

// ---- tiny markdown renderer (code blocks, inline code, bold, lists) ----
function mdToHtml(t) {
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const parts = t.split(/```/);
  let out = "";
  parts.forEach((seg, i) => {
    if (i % 2 === 1) {
      out += `<pre><code>${esc(seg.replace(/^\w*\n/, ""))}</code></pre>`;
    } else {
      let h = esc(seg)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
        .replace(/^\s*[-*]\s+(.*)$/gm, "• $1")
        .replace(/\n/g, "<br>");
      out += h;
    }
  });
  return out;
}

function addMsg(role, text, imgUrl) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  if (imgUrl) {
    const im = document.createElement("img");
    im.src = imgUrl;
    el.appendChild(im);
  }
  const body = document.createElement("div");
  body.className = "body";
  body.innerHTML = mdToHtml(text);
  el.appendChild(body);
  $("chat").appendChild(el);
  $("chat").scrollTop = $("chat").scrollHeight;
  return body;
}

// Ovis grounding: <box>(x1,y1),(x2,y2)</box>. Ovis2.5 emits 0..1 decimals, but
// some prompts yield 0..1000 integers — auto-detect by magnitude.
function parseBoxes(text) {
  const N = "(\\d+(?:\\.\\d+)?)";
  const re = new RegExp(
    `<box>\\(?\\s*${N}\\s*,\\s*${N}\\s*\\)?\\s*,\\s*\\(?\\s*${N}\\s*,\\s*${N}\\s*\\)?</box>`,
    "g"
  );
  const boxes = [];
  let m;
  while ((m = re.exec(text))) {
    let c = [+m[1], +m[2], +m[3], +m[4]];
    const scale = Math.max(...c) > 1.5 ? 1000 : 1;
    boxes.push(c.map((n) => n / scale));
  }
  return boxes;
}

// Render the captured shot with highlight rectangles for grounded boxes.
function groundingHtml(shotUrl, boxes) {
  if (!shotUrl || !boxes.length) return "";
  const rects = boxes
    .map(
      ([x1, y1, x2, y2]) =>
        `<div class="gbox" style="left:${x1 * 100}%;top:${y1 * 100}%;` +
        `width:${(x2 - x1) * 100}%;height:${(y2 - y1) * 100}%"></div>`
    )
    .join("");
  return `<div class="ground"><img src="${shotUrl}">${rects}</div>`;
}

// Split streamed text into <think> reasoning and the visible answer.
function renderBotInto(body, full, shotUrl) {
  const m = full.match(/<think>([\s\S]*?)(<\/think>|$)/);
  let think = "", answer = full;
  if (m) {
    think = m[1];
    answer = full.slice(m.index + m[0].length);
  }
  let html = "";
  if (think.trim()) {
    html += `<details class="think"><summary>💭 reasoning</summary>` +
      `<div class="think-body">${mdToHtml(think.trim())}</div></details>`;
  }
  html += mdToHtml(answer.trim());
  html += groundingHtml(shotUrl, parseBoxes(answer));
  body.innerHTML = html;
  $("chat").scrollTop = $("chat").scrollHeight;
}

// Keep only the visible answer (drop <think>…</think>) when storing history.
function stripThink(t) {
  const m = t.match(/<think>[\s\S]*?<\/think>/);
  return (m ? t.slice(m.index + m[0].length) : t).trim();
}

// ---- new chat: wipe this session's context ----
function newChat() {
  conversation = [];
  $("chat").innerHTML = "";
  clearShot();
}
$("newchat").addEventListener("click", newChat);

// ---- send + stream ----
async function send() {
  const text = $("prompt").value.trim();
  if (!text) return;
  $("send").disabled = true;
  $("prompt").value = "";

  const content = [];
  if (shot) content.push({ type: "image_url", image_url: { url: shot } });
  content.push({ type: "text", text });
  addMsg("user", text, shot);
  const capturedShot = shot;
  clearShot();

  // Append this turn to the running conversation and send the WHOLE history.
  const userTurn = { role: "user", content };
  conversation.push(userTurn);

  const body = addMsg("bot", "…");
  let full = "";
  try {
    const resp = await fetch(`${API}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: conversation,
        max_tokens: 768,
        temperature: 0.0,
        stream: true,
        enable_thinking: $("deep").checked,
        enable_thinking_budget: $("deep").checked,
        thinking_budget: 1024,
      }),
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const d = line.slice(6).trim();
        if (d === "[DONE]") continue;
        try {
          const j = JSON.parse(d);
          const delta = j.choices?.[0]?.delta?.content || "";
          if (delta) { full += delta; renderBotInto(body, full, capturedShot); }
        } catch {}
      }
    }
    if (!full) {
      body.textContent = "(empty response)";
      conversation.pop(); // don't leave a user turn with no reply
    } else {
      // Record the assistant's answer so the next turn has context.
      conversation.push({ role: "assistant", content: stripThink(full) });
    }
  } catch (e) {
    body.textContent = "Request failed — is the daemon running on :8000? " + e.message;
    conversation.pop(); // roll back the failed user turn
  } finally {
    $("send").disabled = false;
  }
}

$("send").addEventListener("click", send);
$("prompt").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
