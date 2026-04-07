const pptxgen = require("pptxgenjs");

// ─── Color palette ────────────────────────────────────────────────────────────
const BG       = "0D1B2A";   // dark navy background
const TEAL     = "00C9A7";   // accent / highlight
const TEALDK   = "008F77";   // darker teal for table headers
const WHITE    = "FFFFFF";
const LTBLUE   = "8EC8E8";   // secondary text
const CARD     = "152232";   // card / box fill
const ROWEVEN  = "0F2035";
const ROWODD   = "132840";
const YELLOW   = "FFD166";
const GRAY     = "7B9DAA";

// ─── Dimensions (LAYOUT_WIDE = 13.3" × 7.5") ─────────────────────────────────
const W = 13.3;
const H = 7.5;

let pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title  = "Week 08: The World of Large Language Models";
pres.author = "Professor Antonio de Pádua Paes Jr.";

// ─── Helpers ──────────────────────────────────────────────────────────────────
function ns(bgColor) {
  let s = pres.addSlide();
  s.background = { color: bgColor || BG };
  return s;
}

function topBar(slide, color) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: W, h: 0.07,
    fill: { color: color || TEAL },
    line: { color: color || TEAL }
  });
}

function slideTitle(slide, text, y, color, size) {
  slide.addText(text, {
    x: 0.5, y: y || 0.25, w: W - 1, h: 0.72,
    fontSize: size || 30, bold: true,
    color: color || WHITE, fontFace: "Calibri",
    margin: 0
  });
}

function sub(slide, text, y, color) {
  slide.addText(text, {
    x: 0.5, y: y || 1.0, w: W - 1, h: 0.4,
    fontSize: 17, color: color || TEAL,
    fontFace: "Calibri", bold: false, margin: 0
  });
}

function body(slide, text, x, y, w, h, opts) {
  opts = opts || {};
  slide.addText(text, {
    x: x, y: y, w: w, h: h,
    fontSize: opts.size || 14, color: opts.color || WHITE,
    fontFace: "Calibri", valign: opts.valign || "top",
    bold: opts.bold || false, italic: opts.italic || false,
    align: opts.align || "left", margin: opts.margin !== undefined ? opts.margin : 0
  });
}

function bullets(slide, items, x, y, w, h, opts) {
  opts = opts || {};
  const arr = items.map((item, i) => ({
    text: typeof item === "string" ? item : item.text,
    options: {
      bullet: true,
      indentLevel: (typeof item === "object" && item.indent) ? item.indent : 0,
      bold: typeof item === "object" && item.bold ? true : false,
      color: (typeof item === "object" && item.color) ? item.color : (opts.color || WHITE),
      fontSize: (typeof item === "object" && item.size) ? item.size : (opts.size || 14),
      breakLine: i < items.length - 1
    }
  }));
  slide.addText(arr, {
    x: x, y: y, w: w, h: h,
    fontFace: "Calibri", valign: "top", margin: 0
  });
}

// Standard table helper: headers + data rows (alternating row colors)
function tbl(slide, headers, rows, x, y, w, h, colW, opts) {
  opts = opts || {};
  const hdrRow = headers.map(h => ({
    text: h,
    options: {
      fill: { color: opts.hdrFill || TEALDK },
      color: WHITE, bold: true,
      fontSize: opts.hdrSize || 11,
      align: "center", valign: "middle"
    }
  }));
  const dataRows = rows.map((row, ri) =>
    row.map(cell => ({
      text: cell || "",
      options: {
        fill: { color: ri % 2 === 0 ? ROWEVEN : ROWODD },
        color: (typeof cell === "object" && cell.color) ? cell.color : WHITE,
        fontSize: opts.rowSize || 10,
        valign: "middle",
        bold: false
      }
    }))
  );
  const tableOpts = { x, y, w, h, fontFace: "Calibri",
    border: { pt: 0.5, color: "1A3A52" } };
  if (colW) tableOpts.colW = colW;
  slide.addTable([hdrRow, ...dataRows], tableOpts);
}

// Colored stat card
function card(slide, x, y, w, h, title, value, sub2) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: CARD },
    line: { color: TEAL, pt: 1.5 }
  });
  slide.addText(value, {
    x: x + 0.1, y: y + 0.1, w: w - 0.2, h: h * 0.55,
    fontSize: 28, bold: true, color: TEAL,
    fontFace: "Calibri", align: "center", valign: "middle", margin: 0
  });
  slide.addText(title, {
    x: x + 0.1, y: y + h * 0.58, w: w - 0.2, h: h * 0.26,
    fontSize: 11, bold: true, color: WHITE,
    fontFace: "Calibri", align: "center", margin: 0
  });
  if (sub2) {
    slide.addText(sub2, {
      x: x + 0.1, y: y + h * 0.82, w: w - 0.2, h: h * 0.18,
      fontSize: 9, color: LTBLUE,
      fontFace: "Calibri", align: "center", margin: 0
    });
  }
}

// ─── SLIDE 1: Title ───────────────────────────────────────────────────────────
{
  let s = ns("0A1628");
  // Big teal accent bar top
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:W, h:0.12, fill:{color:TEAL}, line:{color:TEAL} });
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:H-0.12, w:W, h:0.12, fill:{color:TEAL}, line:{color:TEAL} });
  // Decorative side accent
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0.12, w:0.06, h:H-0.24, fill:{color:"00A88C"}, line:{color:"00A88C"} });
  // Title
  s.addText("The World of", {
    x:0.5, y:1.4, w:12.3, h:1.1,
    fontSize:52, bold:true, color:LTBLUE, fontFace:"Calibri", align:"center", margin:0
  });
  s.addText("Large Language Models", {
    x:0.5, y:2.5, w:12.3, h:1.2,
    fontSize:52, bold:true, color:TEAL, fontFace:"Calibri", align:"center", margin:0
  });
  s.addText("BUAN 6v99 — Generative AI for Business  |  Class 8", {
    x:0.5, y:3.9, w:12.3, h:0.5,
    fontSize:20, color:WHITE, fontFace:"Calibri", align:"center", margin:0
  });
  s.addText("University of Texas at Dallas  |  Spring 2026", {
    x:0.5, y:4.45, w:12.3, h:0.4,
    fontSize:16, color:LTBLUE, fontFace:"Calibri", align:"center", margin:0
  });
  s.addText("Professor: Antonio de Pádua Paes Jr.", {
    x:0.5, y:5.0, w:12.3, h:0.35,
    fontSize:14, color:GRAY, fontFace:"Calibri", italic:true, align:"center", margin:0
  });
  s.addText('"Language is the interface between human intelligence and machine intelligence.\nLarge Language Models are making that interface seamless."', {
    x:1.5, y:5.55, w:10.3, h:0.85,
    fontSize:13, color:GRAY, fontFace:"Calibri", italic:true, align:"center", margin:0
  });
}

// ─── SLIDE 2: Agenda ──────────────────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Today's Agenda", 0.25);
  sub(s, "Class Overview — 2 Hours", 1.0);
  tbl(s,
    ["#", "Section", "Time"],
    [
      ["1", "What is an LLM? — Foundation", "10 min"],
      ["2", "The History of LLMs — From 2017 to Today", "25 min"],
      ["3", "Top Paid LLMs — ChatGPT, Claude, Gemini, Perplexity", "20 min"],
      ["4", "Free & Open-Source LLMs — Llama, Mistral, Ollama", "20 min"],
      ["5", "The Tools Ecosystem — What's Built on Top", "25 min"],
      ["6", "Use Cases — Choosing the Right LLM for Your Business", "20 min"],
    ],
    0.5, 1.55, 12.3, 4.0, [0.6, 9.5, 2.2],
    { hdrSize: 12, rowSize: 13 }
  );
  body(s, "Wrap-up: Key takeaways + 8 hands-on exercises you'll complete this week",
    0.5, 5.7, 12.3, 0.5, { color: LTBLUE, size: 13, italic: true });
}

// ─── SLIDE 3: What is an LLM? ─────────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "What Is a Large Language Model?");
  sub(s, "The Foundation", 1.0);
  body(s, "A Large Language Model (LLM) is an AI system trained on massive amounts of text data to understand and generate human language.",
    0.5, 1.45, 12.3, 0.65, { size: 15, color: LTBLUE });
  tbl(s,
    ["Characteristic", "What It Means"],
    [
      ["Large", "Billions to trillions of parameters — weights learned during training"],
      ["Language", "Operates on text — reads, understands, generates, translates, summarizes"],
      ["Model", "A mathematical system that predicts what text should come next"],
    ],
    0.5, 2.2, 12.3, 1.85, [2.2, 10.1]
  );
  body(s, "How it works (simplified):", 0.5, 4.2, 12.3, 0.35, { color: TEAL, bold: true, size: 14 });
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:4.6, w:12.3, h:0.6, fill:{color:CARD}, line:{color:TEAL, pt:1} });
  body(s, 'You give it text  →  it predicts the most likely continuation  →  that\'s its "answer"',
    0.5, 4.65, 12.3, 0.5, { color: WHITE, size: 14, align: "center" });
  body(s, "Why it matters for business:", 0.5, 5.35, 5, 0.35, { color: TEAL, bold: true, size: 13 });
  bullets(s,
    ["Can perform tasks that previously required expensive human expertise",
     "Writing, analysis, coding, research, customer service — all at scale",
     "No programming knowledge required to use"],
    0.5, 5.75, 12.3, 1.4, { size: 13 }
  );
}

// ─── SLIDE 4: History 2017–2019 ───────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "The History of LLMs — 2017–2019: The Spark");
  sub(s, "The Transformer Revolution", 1.0);
  const years = [
    { yr: "2017", title: '"Attention Is All You Need"  (Google Brain)',
      pts: ["Landmark paper introducing the Transformer architecture",
            "Key innovation: self-attention — which words relate to which other words",
            "Every major LLM today is built on this architecture"] },
    { yr: "2018", title: "BERT (Google)",
      pts: ["Bidirectional Encoder — reads text in both directions simultaneously",
            "Revolutionized search engines — Google still uses it today"] },
    { yr: "2018", title: "GPT-1 (OpenAI)",
      pts: ["117 million parameters; could generate coherent paragraphs for the first time"] },
    { yr: "2019", title: "GPT-2 (OpenAI)",
      pts: ["1.5 billion parameters — 10× bigger than GPT-1",
            "So capable OpenAI initially refused to release it publicly"] },
  ];
  let yPos = 1.42;
  years.forEach(entry => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:yPos, w:1.1, h:0.35,
      fill:{color:TEAL}, line:{color:TEAL} });
    s.addText(entry.yr, { x:0.5, y:yPos, w:1.1, h:0.35,
      fontSize:12, bold:true, color:"0D1B2A", fontFace:"Calibri", align:"center", valign:"middle", margin:0 });
    body(s, entry.title, 1.75, yPos, 10.8, 0.35, { size:13, bold:true, color:WHITE });
    yPos += 0.38;
    entry.pts.forEach(pt => {
      bullets(s, [pt], 1.75, yPos, 10.8, 0.32, { size:12, color:LTBLUE });
      yPos += 0.30;
    });
    yPos += 0.10;
  });
}

// ─── SLIDE 5: History 2020–2022 ───────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "The History of LLMs — 2020–2022: The Scaling Era");
  sub(s, "When Size Became a Superpower", 1.0);
  const items = [
    { yr:"2020", title:"GPT-3 (OpenAI) — 175 billion parameters, 100× leap from GPT-2",
      pts:["Few-shot learning: give 3 examples and it learns the task",
           "Powered early ChatGPT prototypes, sparked the AI startup boom"] },
    { yr:"2021", title:"Codex (OpenAI) — GPT-3 fine-tuned on code",
      pts:["Became the engine behind GitHub Copilot"] },
    { yr:"2021", title:"DALL-E (OpenAI) — Text-to-image generation",
      pts:["Launched the multimodal AI era"] },
    { yr:"2022", title:"InstructGPT / RLHF — Reinforcement Learning from Human Feedback",
      pts:["Models learned to follow instructions and be helpful — usable by non-experts"] },
    { yr:"Nov 2022", title:"ChatGPT — First public consumer LLM",
      pts:["1 million users in 5 days — fastest product adoption in history",
           "Changed the public conversation about AI forever"] },
  ];
  let yPos = 1.42;
  items.forEach(entry => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:yPos, w:1.3, h:0.32,
      fill:{color:TEAL}, line:{color:TEAL} });
    s.addText(entry.yr, { x:0.5, y:yPos, w:1.3, h:0.32,
      fontSize:10, bold:true, color:"0D1B2A", fontFace:"Calibri", align:"center", valign:"middle", margin:0 });
    body(s, entry.title, 1.95, yPos, 10.8, 0.32, { size:12, bold:true, color:WHITE });
    yPos += 0.34;
    entry.pts.forEach(pt => {
      bullets(s, [pt], 1.95, yPos, 10.8, 0.28, { size:11, color:LTBLUE });
      yPos += 0.28;
    });
    yPos += 0.07;
  });
}

// ─── SLIDE 6: History 2023–2025 ───────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "The History of LLMs — 2023–2025: The Modern Era");
  sub(s, "Competition, Open Source & Multimodal AI", 1.0);
  body(s, "2023 — The Year of LLMs", 0.5, 1.42, 5.9, 0.35, { color:TEAL, bold:true, size:14 });
  tbl(s,
    ["Event", "Impact"],
    [
      ["GPT-4 (OpenAI, Mar 2023)", "Passed bar exam, SAT — near-human expert performance"],
      ["Claude 1 & 2 (Anthropic)", "Safety-focused; 100K token context window"],
      ["Google Bard → Gemini", "Google's response to ChatGPT; integrated into Workspace"],
      ["Llama 1 & 2 (Meta, Jul 2023)", "Open-source weights — anyone could run LLMs locally"],
      ["Mistral 7B (Sep 2023)", "Small but powerful European open-source model"],
    ],
    0.5, 1.8, 12.3, 1.95, [3.8, 8.5]
  );
  body(s, "2024 — Multimodal & Reasoning", 0.5, 3.85, 5.9, 0.35, { color:TEAL, bold:true, size:14 });
  tbl(s,
    ["Event", "Impact"],
    [
      ["GPT-4o (OpenAI)", 'Voice + vision + text in real-time — the "Her" moment'],
      ["Claude 3 Opus/Sonnet/Haiku", "Outperformed GPT-4 on many benchmarks"],
      ["Llama 3 (Meta)", "Open-source nearly matching proprietary models"],
      ["o1 / o3 (OpenAI)", '"Thinking" models — reasoning before answering'],
    ],
    0.5, 4.22, 12.3, 1.6, [3.8, 8.5]
  );
  body(s, "2025: Agents, local models, domain specialization, price collapse — what cost $20 in 2023 costs $0.10 today",
    0.5, 5.95, 12.3, 0.5, { size:12, color:LTBLUE, italic:true });
}

// ─── SLIDE 7: LLM Market Map ──────────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "The LLM Market Map");
  sub(s, "Navigating the Landscape", 1.0);
  // Two rows: Proprietary top, Open-source bottom
  // Proprietary box
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.42, w:12.5, h:2.45, fill:{color:CARD}, line:{color:"1A3A52",pt:1} });
  body(s, "PROPRIETARY  (Closed Source)", 0.4, 1.42, 12.5, 0.38, { size:11, color:GRAY, bold:true, align:"center" });
  const propCos = [
    { name:"OpenAI", prod:"ChatGPT / GPT-4o", x:1.2 },
    { name:"Anthropic", prod:"Claude", x:4.7 },
    { name:"Google", prod:"Gemini", x:8.2 },
    { name:"Perplexity AI", prod:"Perplexity", x:11.0 },
  ];
  propCos.forEach(co => {
    s.addShape(pres.shapes.RECTANGLE, { x:co.x, y:1.85, w:1.7, h:1.75, fill:{color:"0A1628"}, line:{color:TEAL,pt:1.5} });
    body(s, co.name, co.x, 1.9, 1.7, 0.45, { size:12, bold:true, color:TEAL, align:"center" });
    body(s, co.prod, co.x, 2.35, 1.7, 0.55, { size:11, color:WHITE, align:"center" });
    body(s, "PAID", co.x, 2.9, 1.7, 0.35, { size:10, color:YELLOW, bold:true, align:"center" });
  });
  // Open-source box
  s.addShape(pres.shapes.RECTANGLE, { x:0.4, y:4.1, w:12.5, h:2.55, fill:{color:CARD}, line:{color:"1A3A52",pt:1} });
  body(s, "OPEN-WEIGHT  (Open Source)", 0.4, 4.1, 12.5, 0.38, { size:11, color:GRAY, bold:true, align:"center" });
  const osCos = [
    { name:"Meta AI", prod:"Llama 3", x:1.2 },
    { name:"Mistral AI", prod:"Mistral / Mixtral", x:3.8 },
    { name:"Google", prod:"Gemma", x:6.6 },
    { name:"Microsoft", prod:"Phi-3", x:9.2 },
    { name:"Groq / Ollama", prod:"Any OSS model", x:11.35 },
  ];
  osCos.forEach(co => {
    s.addShape(pres.shapes.RECTANGLE, { x:co.x, y:4.52, w:1.7, h:1.75, fill:{color:"0A1628"}, line:{color:"00A88C",pt:1.5} });
    body(s, co.name, co.x, 4.57, 1.7, 0.45, { size:11, bold:true, color:"00A88C", align:"center" });
    body(s, co.prod, co.x, 5.02, 1.7, 0.55, { size:10, color:WHITE, align:"center" });
    body(s, "FREE", co.x, 5.57, 1.7, 0.35, { size:10, color:TEAL, bold:true, align:"center" });
  });
}

// ─── SLIDE 8: ChatGPT ─────────────────────────────────────────────────────────
{
  let s = ns();
  topBar(s, "10A37F");  // OpenAI green
  slideTitle(s, "OpenAI / ChatGPT — The Market Leader");
  sub(s, "Backed by Microsoft ($13B) · Founded 2015, San Francisco", 1.0, LTBLUE);
  tbl(s,
    ["Model", "Best For", "Context Window", "Speed"],
    [
      ["GPT-4o", "General tasks, vision, voice", "128K tokens", "Fast"],
      ["o3 / o3-mini", "Complex reasoning, math, science", "128K tokens", "Slow (deep thinking)"],
      ["GPT-4.5", "Creative tasks, nuanced writing", "128K tokens", "Fast"],
    ],
    0.5, 1.45, 12.3, 1.6, [2.5,5.0,2.6,2.2]
  );
  body(s, "Pricing", 0.5, 3.15, 3, 0.35, { color:TEAL, bold:true, size:14 });
  bullets(s,
    ["Free tier — ChatGPT with GPT-4o-mini, limited GPT-4o",
     "ChatGPT Plus — $20/month — full GPT-4o, image generation",
     "ChatGPT Pro — $200/month — o1 Pro, unlimited usage",
     "API — Pay per token (~$2.50–$15 per million input tokens)"],
    0.5, 3.55, 5.9, 2.1, { size:13 }
  );
  body(s, "Key Strengths", 6.7, 3.15, 5.9, 0.35, { color:TEAL, bold:true, size:14 });
  bullets(s,
    ["Largest ecosystem — GPT Store, plugins, integrations",
     "Best multimodal capabilities — text, image, voice, video",
     "Widest enterprise adoption and third-party support"],
    6.7, 3.55, 6.1, 1.5, { size:13 }
  );
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:5.75, w:12.3, h:0.55, fill:{color:"1A2B1A"}, line:{color:"CC6600",pt:1} });
  body(s, "⚠  Privacy: your data may train future models without an Enterprise plan",
    0.6, 5.8, 12.1, 0.45, { size:12, color:"FFB347" });
}

// ─── SLIDE 9: Claude ──────────────────────────────────────────────────────────
{
  let s = ns();
  topBar(s, "CC785C");  // Anthropic orange-ish
  slideTitle(s, "Anthropic / Claude — The Thoughtful Alternative");
  sub(s, "Founded 2021 by ex-OpenAI team · Backed by Amazon ($4B) & Google ($300M)", 1.0, LTBLUE);
  tbl(s,
    ["Model", "Best For", "Context Window", "Speed"],
    [
      ["Claude 4 Opus", "Deep analysis, complex reasoning", "200K tokens", "Medium"],
      ["Claude 4 Sonnet", "Balanced: capable + fast + affordable", "200K tokens", "Fast"],
      ["Claude 3.5 Haiku", "Quick tasks, high-volume applications", "200K tokens", "Very fast"],
    ],
    0.5, 1.45, 12.3, 1.6, [2.5,5.0,2.6,2.2]
  );
  body(s, "Key Strengths", 0.5, 3.15, 5.9, 0.35, { color:TEAL, bold:true, size:14 });
  bullets(s,
    [{text:"Longest context window — 200K tokens (≈150,000 words)", bold:true},
     "Best for long documents — contracts, research papers, codebases",
     "Superior instruction following — does exactly what you ask",
     "Projects — persistent memory across conversations",
     "Safety-focused — less likely to produce harmful outputs"],
    0.5, 3.55, 5.9, 2.5, { size:13 }
  );
  body(s, "Pricing", 6.7, 3.15, 5.9, 0.35, { color:TEAL, bold:true, size:14 });
  bullets(s,
    ["Free tier — Claude.ai, limited daily messages",
     "Claude Pro — $20/month, priority access, 5× usage",
     "Claude Team — $30/user/month, shared workspace",
     "API — $3–$15 per million input tokens"],
    6.7, 3.55, 6.1, 2.0, { size:13 }
  );
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:5.75, w:12.3, h:0.55, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, "Watch out for: Smaller ecosystem vs. OpenAI; fewer native integrations",
    0.6, 5.8, 12.1, 0.45, { size:12, color:LTBLUE, italic:true });
}

// ─── SLIDE 10: Gemini ─────────────────────────────────────────────────────────
{
  let s = ns();
  topBar(s, "4285F4");  // Google blue
  slideTitle(s, "Google Gemini — The Integrated Giant");
  sub(s, "Google DeepMind · Deep integration with Gmail, Docs, Sheets, Slides, Meet", 1.0, LTBLUE);
  tbl(s,
    ["Model", "Best For", "Context Window", "Speed"],
    [
      ["Gemini Ultra", "Most complex tasks, research", "1M tokens", "Slow"],
      ["Gemini Pro 1.5", "Balanced everyday tasks", "1M tokens", "Medium"],
      ["Gemini Flash", "High speed, cost-efficient", "1M tokens", "Very fast"],
      ["Gemini Nano", "On-device, mobile (Pixel phones)", "Small", "Ultra fast"],
    ],
    0.5, 1.45, 12.3, 1.8, [2.5,5.0,2.6,2.2]
  );
  body(s, "Key Strengths", 0.5, 3.35, 5.9, 0.35, { color:TEAL, bold:true, size:14 });
  bullets(s,
    [{text:"1 million token context window — largest of any mainstream model", bold:true},
     "Native multimodal — built for text, images, audio, video",
     "Seamless in Gmail, Docs, Drive, Sheets, Slides, Meet",
     "NotebookLM — extraordinary tool for research and learning",
     "Real-time search — can access current web information"],
    0.5, 3.75, 5.9, 2.6, { size:13 }
  );
  body(s, "Pricing", 6.7, 3.35, 5.9, 0.35, { color:TEAL, bold:true, size:14 });
  bullets(s,
    ["Free tier — Gemini.google.com, Gemini Pro access",
     "Google One AI Premium — $20/month, Gemini Ultra",
     "Workspace Business — $30/user/month, AI in all apps",
     "API — via Google AI Studio (free tier available)"],
    6.7, 3.75, 6.1, 2.0, { size:13 }
  );
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:5.75, w:12.3, h:0.55, fill:{color:CARD}, line:{color:"4285F4",pt:1} });
  body(s, "Watch out for: Still catching up on pure text reasoning vs. OpenAI/Anthropic",
    0.6, 5.8, 12.1, 0.45, { size:12, color:LTBLUE, italic:true });
}

// ─── SLIDE 11: Perplexity ─────────────────────────────────────────────────────
{
  let s = ns();
  topBar(s, "20B2AA");
  slideTitle(s, "Perplexity AI — Search Reimagined");
  sub(s, "Founded 2022 · Backed by Amazon, NVIDIA, Jeff Bezos · AI-powered search engine", 1.0, LTBLUE);
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:1.48, w:12.3, h:0.62, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, '"Every answer comes with sources. Every claim is verifiable."',
    0.6, 1.55, 12.1, 0.48, { size:15, italic:true, color:TEAL, align:"center" });
  tbl(s,
    ["Feature", "Why It Matters"],
    [
      ["Always Up-to-Date", "Searches the web in real time — no knowledge cutoff"],
      ["Source Citations", "Every answer links to original, verifiable sources"],
      ["Research Mode", "Deep multi-step research on complex topics"],
      ["Academic Mode", "Searches peer-reviewed papers (PubMed, arXiv)"],
      ["Business Use", "Market research, competitor analysis, trend tracking"],
    ],
    0.5, 2.2, 12.3, 2.3, [3.0, 9.3]
  );
  body(s, "Pricing:", 0.5, 4.65, 2.5, 0.35, { color:TEAL, bold:true, size:14 });
  bullets(s, ["Free tier — unlimited searches, limited Pro searches",
    "Perplexity Pro — $20/month, unlimited Pro searches, image gen, file uploads"],
    0.5, 5.05, 8, 0.9, { size:13 });
  body(s, "Best for: Market research, fact-checking, current events, academic research",
    0.5, 6.0, 12.3, 0.4, { size:13, color:LTBLUE });
  body(s, "Not ideal for: Creative writing, long-form generation, conversational tasks",
    0.5, 6.45, 12.3, 0.4, { size:13, color:GRAY });
}

// ─── SLIDE 12: Paid LLM Comparison ───────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Paid LLMs — Business Comparison");
  sub(s, "Choosing the Right Paid Tool", 1.0);
  tbl(s,
    ["", "ChatGPT (OpenAI)", "Claude (Anthropic)", "Gemini (Google)", "Perplexity"],
    [
      ["Best For", "General tasks, images, voice", "Long docs, analysis, coding", "Google Workspace users", "Research, fact-checking"],
      ["Context Window", "128K tokens", "200K tokens", "1M tokens", "Real-time web"],
      ["Starting Price", "$20/month", "$20/month", "$20/month", "$20/month"],
      ["Free Tier", "Yes (limited)", "Yes (limited)", "Yes (limited)", "Yes (good)"],
      ["Key Strength", "Ecosystem, multimodal", "Long context, precision", "Google integration", "Real-time accuracy"],
      ["Privacy", "⚠ Trains on data", "✓ No training by default", "⚠ Google data", "✓ Sources cited"],
      ["Enterprise Plan", "$30/user/month", "$30/user/month", "$30/user/month", "$40/user/month"],
    ],
    0.5, 1.45, 12.3, 3.8, [2.0,2.6,2.6,2.6,2.5],
    { hdrSize:11, rowSize:10 }
  );
  body(s, "Quick Decision Guide:", 0.5, 5.45, 4, 0.35, { color:TEAL, bold:true, size:13 });
  bullets(s,
    ["Long document analysis → Claude (200K context)",
     "Images + voice + video → ChatGPT (GPT-4o)",
     "Google Workspace heavy user → Gemini",
     "Research + current events → Perplexity",
     "Complex math/science reasoning → OpenAI o3"],
    0.5, 5.85, 12.3, 1.5, { size:12, color:LTBLUE }
  );
}

// ─── SLIDE 13: What Does "Free" Mean? ────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, 'What Does "Free" Really Mean?');
  sub(s, "Three Very Different Things", 1.0);
  const types = [
    { num:"1", title:"Free API Tier — Still proprietary, just free up to a limit",
      pts:["Examples: Gemini API free tier, Claude.ai free plan, OpenAI free tier",
           "Usage limits apply — rate limits, message caps; data may train models"] },
    { num:"2", title:"Open-Weight Models — The model weights are public",
      pts:["Examples: Meta Llama 3, Mistral 7B, Google Gemma",
           "Download and run yourself — on your computer or cloud; full control to fine-tune"] },
    { num:"3", title:"Free Hosted Interfaces — Open-weight models on someone else's server",
      pts:["Examples: Ollama (local), Groq (cloud), Hugging Face Spaces",
           "Free open model without the setup hassle"] },
  ];
  let yPos = 1.45;
  types.forEach(t => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:yPos, w:0.55, h:0.55, fill:{color:TEAL}, line:{color:TEAL} });
    body(s, t.num, 0.5, yPos, 0.55, 0.55, { size:18, bold:true, color:"0D1B2A", align:"center", valign:"middle" });
    body(s, t.title, 1.2, yPos, 11.5, 0.38, { size:13, bold:true, color:WHITE });
    yPos += 0.42;
    t.pts.forEach(pt => {
      bullets(s, [pt], 1.2, yPos, 11.5, 0.3, { size:12, color:LTBLUE });
      yPos += 0.30;
    });
    yPos += 0.14;
  });
  // Mini table
  tbl(s,
    ["Concern", "Free API Tier", "Open-Weight", "Free Hosted"],
    [
      ["Data Privacy", "⚠ Shared with provider", "✓ 100% private", "✓ Mostly private"],
      ["Cost at Scale", "Paid beyond limits", "Infra cost only", "Free or very cheap"],
      ["Customization", "✗ Limited", "✓ Full fine-tuning", "Limited"],
    ],
    0.5, 5.75, 12.3, 1.5, [2.5,3.26,3.26,3.28]
  );
}

// ─── SLIDE 14: Llama 3 ────────────────────────────────────────────────────────
{
  let s = ns();
  topBar(s, "0768FC");  // Meta blue
  slideTitle(s, "Meta Llama 3 — Open Source Champion");
  sub(s, "Open weights — free to download, use, and modify · MIT-style license", 1.0, LTBLUE);
  tbl(s,
    ["Model", "Parameters", "Best For", "Can Run On"],
    [
      ["Llama 3.2 1B / 3B", "1–3 billion", "Mobile, edge devices", "Phones, Raspberry Pi"],
      ["Llama 3.2 11B / 90B", "11–90 billion", "General tasks, vision", "Gaming PC, Mac M3"],
      ["Llama 3.1 405B", "405 billion", "Near-GPT-4 quality", "Data center / cloud"],
    ],
    0.5, 1.48, 12.3, 1.65, [2.8,2.5,3.5,3.5]
  );
  body(s, "Where to Access Llama 3:", 0.5, 3.25, 5, 0.35, { color:TEAL, bold:true, size:14 });
  tbl(s,
    ["Platform", "Cost", "Privacy", "Ease of Use"],
    [
      ["Meta.ai", "Free", "Meta sees it", "Very easy"],
      ["Ollama (local)", "Free", "100% private", "Easy"],
      ["Groq (cloud API)", "Free tier", "Groq sees it", "Easy"],
      ["Amazon Bedrock", "Pay per use", "Your AWS account", "Medium"],
    ],
    0.5, 3.65, 8.5, 1.8, [2.2,1.8,2.5,2.0]
  );
  body(s, "Business Case for Llama:", 0.5, 5.6, 5, 0.35, { color:TEAL, bold:true, size:14 });
  bullets(s,
    ["Zero per-token cost at scale — huge savings for high-volume applications",
     "Full data privacy — sensitive business data never leaves your servers",
     "Can be fine-tuned on your company's data and terminology"],
    0.5, 5.98, 12.3, 1.2, { size:13 }
  );
}

// ─── SLIDE 15: Mistral ────────────────────────────────────────────────────────
{
  let s = ns();
  topBar(s, "FF7000");  // Mistral orange
  slideTitle(s, "Mistral AI — The European Challenger");
  sub(s, "Founded 2023, Paris · Backed by a16z, Lightspeed, NVIDIA · Quality over size", 1.0, LTBLUE);
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:1.48, w:12.3, h:0.52, fill:{color:CARD}, line:{color:"FF7000",pt:1} });
  body(s, "Mistral 7B outperformed Llama 2 13B on every benchmark — at half the size",
    0.6, 1.54, 12.1, 0.40, { size:14, italic:true, color:WHITE, align:"center" });
  tbl(s,
    ["Model", "Parameters", "Type", "Best For"],
    [
      ["Mistral 7B", "7B", "Open-weight", "Fast, lightweight tasks"],
      ["Mixtral 8×7B", "45B active (MoE)", "Open-weight", "High quality at lower cost"],
      ["Mistral Small", "—", "API", "Cost-efficient API usage"],
      ["Mistral Large", "—", "API", "Complex reasoning, code"],
      ["Codestral", "—", "API", "Code generation (all languages)"],
    ],
    0.5, 2.1, 12.3, 2.2, [2.5,2.5,2.2,5.1]
  );
  body(s, "MoE — Mixture of Experts (Mixtral):", 0.5, 4.42, 7, 0.35, { color:TEAL, bold:true, size:13 });
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:4.82, w:12.3, h:0.52, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, "Activates only the 2 most relevant \"expert\" sub-networks → GPT-3.5 quality at a fraction of compute cost",
    0.6, 4.87, 12.1, 0.42, { size:12, color:WHITE });
  body(s, "Pricing: Open-weight — Free (download and run)  ·  Mistral API free tier  ·  Paid from ~$0.002/1K tokens",
    0.5, 5.45, 12.3, 0.38, { size:12, color:LTBLUE });
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:5.9, w:12.3, h:0.55, fill:{color:CARD}, line:{color:"FF7000",pt:1} });
  body(s, "European advantage: GDPR-compliant by design · Data stored in EU — important for regulated industries",
    0.6, 5.95, 12.1, 0.45, { size:12, color:"FFB347" });
}

// ─── SLIDE 16: Gemma & Phi-3 ─────────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Google Gemma & Microsoft Phi-3 — Small Models, Big Impact");
  // Left column: Gemma
  body(s, "Google Gemma", 0.5, 1.05, 5.9, 0.42, { color:TEAL, bold:true, size:18 });
  body(s, "Released Feb 2024 · Open weights · Built by Google DeepMind", 0.5, 1.5, 5.9, 0.35, { color:LTBLUE, size:11 });
  tbl(s,
    ["Model", "Params", "Special Feature"],
    [
      ["Gemma 2B", "2B", "Runs on phones and laptops"],
      ["Gemma 7B", "7B", "Strong general performance"],
      ["Gemma 2 (9B/27B)", "9–27B", "State-of-the-art for its size"],
      ["CodeGemma", "7B", "Optimized for code"],
    ],
    0.5, 1.9, 5.9, 1.8, [1.8,1.2,2.9]
  );
  body(s, "Best for: Developers embedding Google-quality AI in their own apps, on-device AI",
    0.5, 3.78, 5.9, 0.65, { size:12, color:LTBLUE });
  // Divider
  s.addShape(pres.shapes.RECTANGLE, { x:6.65, y:1.05, w:0.04, h:5.5, fill:{color:"1A3A52"}, line:{color:"1A3A52"} });
  // Right column: Phi-3
  body(s, "Microsoft Phi-3", 6.9, 1.05, 6.0, 0.42, { color:TEAL, bold:true, size:18 });
  body(s, "Released Apr 2024 · Open weights (MIT) · Train on textbook-quality data", 6.9, 1.5, 6.0, 0.35, { color:LTBLUE, size:11 });
  tbl(s,
    ["Model", "Params", "Context", "Remarkable Fact"],
    [
      ["Phi-3 Mini", "3.8B", "128K", "Fits on a phone, beats GPT-3.5"],
      ["Phi-3 Small", "7B", "128K", "Better than Mixtral on reasoning"],
      ["Phi-3 Medium", "14B", "128K", "Approaches GPT-4 on benchmarks"],
    ],
    6.9, 1.9, 6.0, 1.55, [1.5,1.1,1.1,2.3]
  );
  body(s, "Why small models matter for business:", 6.9, 3.55, 6.0, 0.35, { color:TEAL, bold:true, size:13 });
  bullets(s,
    ["Run on-device — no internet connection needed (field workers, secure environments)",
     "Zero latency — response in milliseconds",
     "No per-call cost — deploy once, use forever",
     "Compliance-friendly — data never leaves the device"],
    6.9, 3.95, 6.0, 2.2, { size:12 }
  );
}

// ─── SLIDE 17: Groq & Ollama ──────────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Groq & Ollama — Speed and Privacy");
  sub(s, "Supercharging Open-Source Models", 1.0);
  // Left: Groq
  body(s, "Groq — Inference at the Speed of Thought", 0.5, 1.45, 6.0, 0.45, { color:TEAL, bold:true, size:16 });
  body(s, "Cloud platform running open-source models at extraordinary speed\nusing custom LPU (Language Processing Unit) chips",
    0.5, 1.95, 6.0, 0.65, { size:12, color:LTBLUE });
  tbl(s,
    ["Platform", "Speed (tokens/sec)"],
    [
      ["Standard cloud GPU", "~40 t/s"],
      ["OpenAI API (GPT-4o)", "~60 t/s"],
      ["Groq", "~250–800 t/s  ⚡"],
    ],
    0.5, 2.72, 5.9, 1.55, [3.6, 2.3]
  );
  bullets(s,
    ["Models: Llama 3.1/3.3, Mixtral, Gemma, Whisper (audio)",
     "Pricing: Free tier (rate-limited) · $0.05–$0.79/million tokens",
     "Best for: Real-time apps, voice interfaces, high-throughput"],
    0.5, 4.4, 5.9, 1.4, { size:12 }
  );
  // Right: Ollama
  s.addShape(pres.shapes.RECTANGLE, { x:6.65, y:1.45, w:0.04, h:4.5, fill:{color:"1A3A52"}, line:{color:"1A3A52"} });
  body(s, "Ollama — Your Private AI on Your Laptop", 6.9, 1.45, 6.0, 0.45, { color:TEAL, bold:true, size:16 });
  body(s, "Free tool to run LLMs locally on your Mac, Windows, or Linux machine",
    6.9, 1.95, 6.0, 0.4, { size:12, color:LTBLUE });
  s.addShape(pres.shapes.RECTANGLE, { x:6.9, y:2.45, w:5.9, h:0.65, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, "ollama pull llama3.2   →   ollama run llama3.2",
    6.9, 2.52, 5.9, 0.52, { size:13, color:TEAL, fontFace:"Consolas", align:"center" });
  bullets(s,
    ["Supported: Llama 3, Mistral, Phi-3, Gemma, CodeLlama, 100+ models",
     "Hardware: Mac M1/M2/M3 or GPU PC (8GB+ RAM minimum)",
     "Cost: 100% free — no API keys, no internet after download"],
    6.9, 3.22, 6.0, 1.3, { size:12 }
  );
  body(s, "Business use cases: Sensitive data analysis (HR, legal, financials) · Offline environments · Fast prototyping",
    6.9, 4.62, 6.0, 0.7, { size:12, color:LTBLUE });
}

// ─── SLIDE 18: Free LLM Comparison ───────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Free & Open-Source LLMs — Business Comparison");
  sub(s, "The Free Tier Decision Matrix", 1.0);
  tbl(s,
    ["", "Llama 3 (Meta)", "Mistral/Mixtral", "Gemma (Google)", "Phi-3 (Microsoft)", "Groq", "Ollama"],
    [
      ["Access", "Download / API", "Download / API", "Download / API", "Download / API", "Cloud API", "Local install"],
      ["Best For", "General tasks", "Efficient reasoning", "On-device AI", "Small + smart", "Speed-critical", "Privacy-first"],
      ["Privacy", "✓ Self-hosted", "✓ Self-hosted", "✓ Self-hosted", "✓ On-device", "⚠ Groq cloud", "✓ 100% local"],
      ["Cost", "Free weights", "Free weights", "Free weights", "Free weights", "Free tier", "100% free"],
      ["Setup Effort", "Medium", "Medium", "Medium", "Medium", "Low (API)", "Low"],
      ["Fine-tunable", "✓ Yes", "✓ Yes", "✓ Yes", "✓ Yes", "✗ No", "✓ Yes"],
    ],
    0.5, 1.45, 12.3, 3.2, [1.5,1.7,1.7,1.7,1.8,1.45,1.45],
    { hdrSize:10, rowSize:10 }
  );
  // Two-column decision guide
  body(s, "Go open-source when:", 0.5, 4.8, 5.9, 0.35, { color:TEAL, bold:true, size:13 });
  bullets(s,
    ["Handling sensitive/confidential business data",
     "High volume (millions of API calls/month)",
     "Need to customize/fine-tune on your domain",
     "Budget is a primary constraint"],
    0.5, 5.2, 5.9, 2.0, { size:12 }
  );
  body(s, "Stick with paid when:", 6.7, 4.8, 5.9, 0.35, { color:YELLOW, bold:true, size:13 });
  bullets(s,
    ["Need the absolute best quality available",
     "Speed-to-production matters more than cost",
     "Team lacks technical AI infrastructure expertise",
     "Multimodal (vision, voice) is required"],
    6.7, 5.2, 6.1, 2.0, { size:12 }
  );
}

// ─── SLIDE 19: Tools Ecosystem ────────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "How the Tools Ecosystem Works");
  sub(s, "From Model to Product — The LLM Value Chain", 1.0);
  // Foundation layer
  s.addShape(pres.shapes.RECTANGLE, { x:3.5, y:1.45, w:6.3, h:0.68, fill:{color:CARD}, line:{color:TEAL,pt:1.5} });
  body(s, "FOUNDATION LAYER:  GPT-4o / Claude / Gemini / Llama / Mistral",
    3.5, 1.52, 6.3, 0.54, { size:12, bold:true, color:TEAL, align:"center" });
  // Arrow
  body(s, "↓  APIs & SDKs  ↓", 3.5, 2.18, 6.3, 0.35, { size:12, color:GRAY, align:"center" });
  // Three columns
  const cols = [
    { title:"NATIVE APPS", sub:"(by LLM company)", items:["ChatGPT","Claude.ai","Gemini","NotebookLM"], x:0.5 },
    { title:"3RD PARTY TOOLS", sub:"(independent companies)", items:["Cursor","Notion AI","Jasper","Perplexity"], x:4.8 },
    { title:"ENTERPRISE", sub:"(large software vendors)", items:["Salesforce Einstein","HubSpot AI","MS Copilot 365","Zendesk AI"], x:9.1 },
  ];
  cols.forEach(col => {
    s.addShape(pres.shapes.RECTANGLE, { x:col.x, y:2.55, w:3.8, h:3.5, fill:{color:CARD}, line:{color:"1A3A52",pt:1} });
    body(s, col.title, col.x+0.1, 2.65, 3.6, 0.42, { size:12, bold:true, color:TEAL, align:"center" });
    body(s, col.sub, col.x+0.1, 3.1, 3.6, 0.32, { size:10, color:GRAY, align:"center" });
    col.items.forEach((item, i) => {
      body(s, item, col.x+0.3, 3.5 + i*0.55, 3.2, 0.45, { size:13, color:WHITE });
    });
  });
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:6.2, w:12.3, h:0.62, fill:{color:"0A1628"}, line:{color:TEAL,pt:1} });
  body(s, "Key insight: The LLM is increasingly a commodity. The application layer built on top is where business value is captured.",
    0.6, 6.27, 12.1, 0.48, { size:12, color:WHITE, italic:true });
}

// ─── SLIDE 20: OpenAI Native Ecosystem ───────────────────────────────────────
{
  let s = ns();
  topBar(s, "10A37F");
  slideTitle(s, "OpenAI's Native Ecosystem");
  sub(s, "Strategy: Create the platform, not just the model", 1.0);
  tbl(s,
    ["Product", "What It Does", "Who Uses It"],
    [
      ["ChatGPT", "Conversational AI, web browsing, image generation", "Everyone"],
      ["DALL-E 3", "Text-to-image generation", "Designers, marketers"],
      ["Sora", "Text-to-video generation", "Content creators"],
      ["GPTs Store", "Custom AI assistants you build and share (3M+ created)", "Power users, businesses"],
      ["Assistants API", "Build AI agents with memory + tools + file access", "Developers"],
      ["Canvas", "AI-powered collaborative document editor", "Writers, professionals"],
      ["Azure OpenAI Service", "Enterprise deployment with security & compliance", "Fortune 500, regulated industries"],
    ],
    0.5, 1.45, 12.3, 3.5, [2.5,7.3,2.5],
    { hdrSize:12, rowSize:11 }
  );
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:5.1, w:12.3, h:0.62, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, "GPT Store Model: Anyone can create a custom ChatGPT without coding · 3M+ custom GPTs since launch · businesses publish GPTs for customer service, internal tools, and more",
    0.6, 5.15, 12.1, 0.52, { size:12, color:LTBLUE });
  body(s, "Strategic play: OpenAI is building the operating system for AI — not just a model provider. The GPT Store = their App Store.",
    0.5, 5.85, 12.3, 0.55, { size:12, italic:true, color:GRAY });
}

// ─── SLIDE 21: Anthropic & Google Ecosystems ─────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Anthropic & Google Ecosystems — Two Philosophies");
  // Left: Anthropic
  body(s, "Anthropic / Claude", 0.5, 1.05, 5.9, 0.45, { color:TEAL, bold:true, size:17 });
  body(s, "Philosophy: Fewer products, done with exceptional quality", 0.5, 1.52, 5.9, 0.35, { color:LTBLUE, size:11, italic:true });
  tbl(s,
    ["Product", "What It Does"],
    [
      ["Claude.ai", "Main chat interface — free and Pro tiers"],
      ["Projects", "Persistent memory workspace across sessions"],
      ["Artifacts", "Generate and edit docs, code, diagrams live in chat"],
      ["Claude API", "Developer access to all models"],
      ["Claude Enterprise", "SSO, admin controls, zero data retention"],
    ],
    0.5, 1.9, 5.9, 2.3, [2.1,3.8]
  );
  body(s, "Claude Sonnet is now the most popular model in Cursor and a key part of GitHub Copilot — a major third-party ecosystem win.",
    0.5, 4.3, 5.9, 0.7, { size:11, color:LTBLUE, italic:true });
  // Divider
  s.addShape(pres.shapes.RECTANGLE, { x:6.65, y:1.05, w:0.04, h:5.5, fill:{color:"1A3A52"}, line:{color:"1A3A52"} });
  // Right: Google
  body(s, "Google's Ecosystem", 6.9, 1.05, 6.0, 0.45, { color:"4285F4", bold:true, size:17 });
  body(s, "Philosophy: Integrate AI everywhere Google already is", 6.9, 1.52, 6.0, 0.35, { color:LTBLUE, size:11, italic:true });
  tbl(s,
    ["Product", "What It Does"],
    [
      ["Gemini", "Main chat interface"],
      ["NotebookLM", "AI research assistant for YOUR documents"],
      ["AI Studio", "Developer playground for Gemini"],
      ["Vertex AI", "Enterprise AI platform on Google Cloud"],
      ["Gemini in Workspace", "AI in Gmail, Docs, Sheets, Slides, Meet"],
      ["Google Search AI", "AI Overviews in search results"],
    ],
    6.9, 1.9, 6.0, 2.55, [2.1,3.9]
  );
  s.addShape(pres.shapes.RECTANGLE, { x:6.9, y:4.55, w:6.0, h:1.15, fill:{color:CARD}, line:{color:"4285F4",pt:1} });
  body(s, "NotebookLM standout:", 7.0, 4.62, 5.8, 0.35, { color:TEAL, bold:true, size:12 });
  bullets(s,
    ["Upload PDFs, YouTube, Google Docs, websites (up to 50 sources)",
     "Generates summaries, FAQs, timelines — and an Audio Podcast of your documents",
     "Business use: onboard employees, analyze reports, create training materials"],
    7.0, 5.02, 5.8, 0.65, { size:10 }
  );
}

// ─── SLIDE 22: Productivity & Writing Tools ───────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Productivity & Writing Tools");
  sub(s, "The Content Creation Layer — $1.8B market · every major writing platform is adding AI", 1.0, LTBLUE);
  tbl(s,
    ["Tool", "Primary LLM", "Category", "Best For", "Price"],
    [
      ["Notion AI", "Claude + GPT-4o", "Docs & notes", "Meeting summaries, wikis, drafts", "$10/user/mo add-on"],
      ["Jasper", "GPT-4o + Claude", "Marketing copy", "Ad copy, blog posts, brand voice", "$49/mo"],
      ["Copy.ai", "GPT-4", "Sales & marketing", "Email sequences, social posts", "Free / $49/mo"],
      ["Grammarly AI", "Proprietary + GPT", "Writing assistant", "Grammar, tone, clarity, rewrites", "Free / $30/mo"],
      ["Quillbot", "Proprietary", "Paraphrasing", "Academic rewriting, summarization", "Free / $10/mo"],
      ["Gamma", "GPT-4o", "Presentations", "Auto-generate slides from a prompt", "Free / $10/mo"],
      ["HeyGen", "Proprietary", "AI Video", "AI avatar video from text script", "$29/mo"],
    ],
    0.5, 1.45, 12.3, 3.8, [1.7,2.2,1.9,4.2,2.3],
    { hdrSize:11, rowSize:10 }
  );
  body(s, "Common pattern: These tools wrap a foundational LLM and add brand voice training, domain templates, workflow integrations, and a simpler interface.",
    0.5, 5.4, 12.3, 0.55, { size:12, color:LTBLUE });
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:6.05, w:12.3, h:0.55, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, "Business reality: These tools cost $10–$50/month each but can save hours of work weekly. ROI is typically positive within weeks.",
    0.6, 6.1, 12.1, 0.45, { size:12, color:WHITE });
}

// ─── SLIDE 23: Developer & Coding Tools ───────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Developer & Coding Tools — AI Enters the IDE");
  sub(s, "GitHub Copilot: 1.8M paying subscribers · developers using AI code 55% faster (GitHub, 2024)", 1.0, LTBLUE);
  tbl(s,
    ["Tool", "LLM Backbone", "Platform", "Key Feature", "Price"],
    [
      ["GitHub Copilot", "Claude + GPT-4o", "VS Code, JetBrains, etc.", "In-editor autocomplete", "$10/mo individual"],
      ["Cursor", "Claude 4 Sonnet", "Standalone IDE", "Full codebase chat + edit", "$20/mo"],
      ["Replit AI", "Custom", "Browser-based IDE", "Run + deploy in the browser", "Free / $25/mo"],
      ["Amazon CodeWhisperer", "Amazon Titan", "AWS ecosystem", "Free for individual devs", "Free"],
      ["Tabnine", "Enterprise-hosted", "Any IDE", "Privacy-first, self-hosted option", "$12/mo"],
      ["Windsurf", "Multiple", "Standalone IDE", "Agentic coding — multi-file edits", "$15/mo"],
    ],
    0.5, 1.45, 12.3, 3.0, [2.0,2.0,2.5,3.5,2.3],
    { hdrSize:11, rowSize:10 }
  );
  body(s, "What these tools can do:", 0.5, 4.55, 5.9, 0.35, { color:TEAL, bold:true, size:13 });
  bullets(s,
    ["Autocomplete entire functions as you type",
     "Explain legacy code you've never seen before",
     "Refactor messy code to best practices",
     "Generate tests automatically",
     "Fix bugs from a plain English error description",
     "Build features from a plain English specification"],
    0.5, 4.95, 5.9, 2.3, { size:12 }
  );
  s.addShape(pres.shapes.RECTANGLE, { x:6.7, y:4.55, w:6.1, h:2.65, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, "Key insight for business students:", 6.8, 4.65, 5.9, 0.35, { color:TEAL, bold:true, size:13 });
  body(s, "You don't need to be a developer to use Cursor or Replit AI.\n\nThese tools make coding accessible to analysts, product managers, and business professionals.",
    6.8, 5.1, 5.9, 2.0, { size:13, color:WHITE });
}

// ─── SLIDE 24: Enterprise Business Tools ──────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Enterprise Business Tools");
  sub(s, "AI Embedded in the Software You Already Use — you may already be paying for it", 1.0, LTBLUE);
  // Three mini tables
  body(s, "CRM & Sales", 0.5, 1.42, 3.9, 0.35, { color:TEAL, bold:true, size:13 });
  tbl(s,
    ["Tool", "AI Feature", "What It Does"],
    [
      ["Salesforce Einstein", "Einstein Copilot", "Summarize calls, draft emails, forecast deals"],
      ["HubSpot AI", "Content Assistant", "Write emails, landing pages, social posts"],
      ["Outreach AI", "Smart sequences", "Personalize cold outreach at scale"],
    ],
    0.5, 1.8, 12.3, 1.4, [2.4,2.2,7.7],
    { hdrSize:10, rowSize:10 }
  );
  body(s, "Customer Service", 0.5, 3.32, 3.9, 0.35, { color:TEAL, bold:true, size:13 });
  tbl(s,
    ["Tool", "AI Feature", "What It Does"],
    [
      ["Zendesk AI", "Intelligent triage", "Auto-categorize tickets, suggest answers"],
      ["Intercom Fin", "AI Agent", "Resolve 50%+ of support tickets without humans"],
      ["Freshdesk Freddy", "Answer Bot", "Deflect routine questions 24/7"],
    ],
    0.5, 3.7, 12.3, 1.35, [2.4,2.2,7.7],
    { hdrSize:10, rowSize:10 }
  );
  body(s, "Productivity & Collaboration", 0.5, 5.17, 4.5, 0.35, { color:TEAL, bold:true, size:13 });
  tbl(s,
    ["Tool", "AI Feature", "What It Does"],
    [
      ["Microsoft Copilot 365", "In Word/Excel/Teams/Outlook", "Summarize meetings, draft docs, analyze data"],
      ["Slack AI", "Channel summaries", "Catch up on long threads instantly"],
      ["Zoom AI Companion", "Meeting intelligence", "Auto-summaries, action items, follow-up emails"],
    ],
    0.5, 5.54, 12.3, 1.4, [2.4,2.8,7.1],
    { hdrSize:10, rowSize:10 }
  );
}

// ─── SLIDE 25: No-Code AI Automation ─────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "No-Code AI Automation");
  sub(s, "Build AI Workflows Without Writing a Single Line of Code", 1.0);
  // n8n highlight box
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:1.42, w:12.3, h:0.95, fill:{color:CARD}, line:{color:TEAL,pt:1.5} });
  body(s, "n8n — The Power User's Platform", 0.6, 1.47, 4, 0.38, { color:TEAL, bold:true, size:13 });
  body(s, "Open-source · self-hostable · 400+ app integrations · AI Agent node with any LLM", 0.6, 1.85, 6, 0.45, { color:LTBLUE, size:11 });
  body(s, "Free (self-hosted) / $20/mo (cloud)", 9.0, 1.55, 3.7, 0.32, { color:YELLOW, bold:true, size:11, align:"right" });
  body(s, "GPT-4o · Claude · Gemini · Mistral · Ollama · any OpenAI-compatible API", 9.0, 1.9, 3.7, 0.38, { color:LTBLUE, size:10, align:"right" });
  // Main table
  tbl(s,
    ["Platform", "LLM Underneath", "Best For", "Price", "Key Differentiator"],
    [
      ["Zapier AI", "GPT-4o (OpenAI)", "Simple automations, beginners", "Free / $29/mo", "Easiest setup, 6,000+ app integrations"],
      ["Make (Integromat)", "GPT-4o, Claude", "Complex multi-branch flows", "Free / $9/mo", "Visual flow designer, very flexible"],
      ["Flowise", "Any (OpenAI, Ollama, HuggingFace)", "Chatbots & RAG pipelines", "Free (open-source)", "Drag-and-drop LangChain flows"],
      ["Dify", "GPT-4o, Claude, Llama, Gemini", "AI app builder for teams", "Free / $59/mo", "Prompt management + RAG + API"],
      ["Relevance AI", "GPT-4o, Claude", "Sales & marketing agents", "Free / $19/mo", "Pre-built agent templates"],
      ["Voiceflow", "GPT-4o, Claude, Gemini", "Conversational AI / chatbots", "Free / $50/mo", "Multi-channel deploy"],
      ["Stack AI", "GPT-4o, Claude, Llama", "Enterprise AI workflows", "Free / $199/mo", "SOC2 compliant, enterprise security"],
    ],
    0.5, 2.45, 12.3, 3.8, [1.8,2.8,2.5,1.8,3.4],
    { hdrSize:11, rowSize:10 }
  );
  body(s, "LLM flexibility advantage: Swap GPT-4o → Claude for cost · → Ollama for 100% private · → Groq for speed — without rebuilding the workflow.",
    0.5, 6.37, 12.3, 0.55, { size:12, color:LTBLUE });
}

// ─── SLIDE 26: Decision Framework ────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Choosing the Right LLM — A Decision Framework");
  sub(s, "Match the Tool to the Task — in 3 steps", 1.0);
  // Step 1
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:1.42, w:0.55, h:0.55, fill:{color:TEAL}, line:{color:TEAL} });
  body(s, "1", 0.5, 1.42, 0.55, 0.55, { size:18, bold:true, color:"0D1B2A", align:"center" });
  body(s, "Identify the task category", 1.2, 1.47, 11, 0.42, { size:14, bold:true, color:WHITE });
  tbl(s,
    ["If you need to…", "Best tool"],
    [
      ["Write / create content", "ChatGPT, Claude, Jasper"],
      ["Research / find current information", "Perplexity, Gemini"],
      ["Analyze long documents / PDFs", "Claude (200K context)"],
      ["Code / build software", "Cursor, Copilot, Claude"],
      ["Work inside Google Workspace", "Gemini + NotebookLM"],
      ["Handle sensitive / private data", "Ollama (local), Claude Enterprise"],
      ["High-volume / cost-sensitive apps", "Groq + Llama, Mistral API"],
    ],
    0.5, 2.05, 9.0, 2.65, [5.5,3.5],
    { hdrSize:11, rowSize:10 }
  );
  // Step 2
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:4.82, w:0.55, h:0.48, fill:{color:TEAL}, line:{color:TEAL} });
  body(s, "2", 0.5, 4.82, 0.55, 0.48, { size:16, bold:true, color:"0D1B2A", align:"center" });
  body(s, "Check budget reality", 1.2, 4.87, 5, 0.38, { size:13, bold:true, color:WHITE });
  bullets(s,
    ["$0/month → Gemini free, Claude free, Meta.ai, Ollama local",
     "$20/month → ChatGPT Plus OR Claude Pro OR Google One AI Premium",
     "Enterprise → Microsoft Copilot 365 + Claude/OpenAI Enterprise"],
    1.2, 5.3, 5.5, 1.4, { size:11 }
  );
  // Step 3
  s.addShape(pres.shapes.RECTANGLE, { x:7.2, y:4.82, w:0.55, h:0.48, fill:{color:TEAL}, line:{color:TEAL} });
  body(s, "3", 7.2, 4.82, 0.55, 0.48, { size:16, bold:true, color:"0D1B2A", align:"center" });
  body(s, "Consider data sensitivity", 7.9, 4.87, 5, 0.38, { size:13, bold:true, color:WHITE });
  bullets(s,
    ["Public / non-sensitive → any cloud LLM is fine",
     "Confidential business data → Claude Enterprise (zero retention) or Ollama",
     "Regulated industry (healthcare, finance, legal) → check HIPAA, SOC2 certs"],
    7.9, 5.3, 5.2, 1.4, { size:11 }
  );
}

// ─── SLIDE 27: Use Case — Content Creation ────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Use Case — Content Creation & Marketing");
  sub(s, "AI Is Now Every Marketer's Co-Writer", 1.0);
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:1.42, w:12.3, h:0.52, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, "Marketing teams using AI produce 3–5× more content at 60% lower cost  (McKinsey, 2024)",
    0.6, 1.49, 12.1, 0.38, { size:13, italic:true, color:TEAL, align:"center" });
  tbl(s,
    ["Content Type", "Best Tool", "Quality Level", "Time Savings"],
    [
      ["Blog posts (1,000–3,000 words)", "ChatGPT + Claude", "High with editing", "70%"],
      ["Social media posts (all platforms)", "Copy.ai, Jasper", "High", "80%"],
      ["Email campaigns & sequences", "HubSpot AI, Jasper", "High", "75%"],
      ["Ad copy (Google, Meta, LinkedIn)", "Jasper, Copy.ai", "High", "80%"],
      ["Product descriptions (e-commerce)", "ChatGPT", "High", "85%"],
      ["Video scripts", "Claude, ChatGPT", "High", "60%"],
      ["Presentation decks", "Gamma, ChatGPT", "Medium", "50%"],
    ],
    0.5, 2.03, 12.3, 3.3, [4.0,2.8,2.5,2.0],
    { hdrSize:11, rowSize:10 }
  );
  body(s, "Best practice workflow:", 0.5, 5.45, 4, 0.35, { color:TEAL, bold:true, size:13 });
  body(s, "Brief AI with brand voice + goal   →   Generate 3 variants   →   Human edits & fact-checks   →   Publish",
    0.5, 5.85, 12.3, 0.45, { size:13, color:WHITE, align:"center" });
  body(s, "Real example: A 5-person team used Claude + Jasper to grow blog output from 4 → 20 posts/month without adding headcount. Organic traffic +140% in 6 months.",
    0.5, 6.4, 12.3, 0.55, { size:11, color:LTBLUE, italic:true });
}

// ─── SLIDE 28: Use Case — Research & Analysis ─────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Use Case — Research & Analysis");
  sub(s, "From Hours to Minutes", 1.0);
  // Before/After comparison
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:1.42, w:5.9, h:0.88, fill:{color:"1A0D0D"}, line:{color:"994444",pt:1} });
  body(s, "Traditional: 4–8 hours", 0.6, 1.47, 5.7, 0.35, { color:"FF8888", bold:true, size:12 });
  body(s, "Search → Read → Synthesize → Write → Cite → Repeat", 0.6, 1.82, 5.7, 0.42, { size:11, color:WHITE });
  s.addShape(pres.shapes.RECTANGLE, { x:6.9, y:1.42, w:5.9, h:0.88, fill:{color:"0D1A0D"}, line:{color:TEAL,pt:1} });
  body(s, "AI-augmented: 30–60 minutes", 7.0, 1.47, 5.7, 0.35, { color:TEAL, bold:true, size:12 });
  body(s, "Perplexity → NotebookLM → Claude analysis → Human validation", 7.0, 1.82, 5.7, 0.42, { size:11, color:WHITE });
  // Tool breakdowns
  const tools = [
    { name:"Perplexity AI", tag:"Start here for current info",
      pts:["Research competitor landscape, industry trends, market size",
           "Always provides sources — easy to verify claims"], color:TEAL },
    { name:"Google NotebookLM", tag:"Your personal research analyst",
      pts:["Upload up to 50 sources: PDFs, videos, articles, Google Docs",
           "Ask questions across all sources simultaneously · generates Audio Podcast"], color:"4285F4" },
    { name:"Claude (200K context)", tag:"Best for long document analysis",
      pts:["Upload entire annual reports, contracts, research papers",
           'Ask: "What are the 5 biggest risks in this document?"'], color:"CC785C" },
    { name:"ChatGPT Advanced Data Analysis", tag:"For structured data",
      pts:["Upload CSV/Excel files and ask AI to chart, summarize, find anomalies"], color:"10A37F" },
  ];
  let yPos = 2.45;
  tools.forEach(tool => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:yPos, w:0.08, h:1.0, fill:{color:tool.color}, line:{color:tool.color} });
    body(s, tool.name, 0.72, yPos, 4.5, 0.38, { size:12, bold:true, color:tool.color });
    body(s, tool.tag, 0.72, yPos+0.38, 4.5, 0.28, { size:10, color:GRAY, italic:true });
    tool.pts.forEach((pt, i) => {
      bullets(s, [pt], 0.72, yPos + 0.66 + i*0.28, 12.2, 0.28, { size:10, color:WHITE });
    });
    yPos += 1.18;
  });
}

// ─── SLIDE 29: Use Case — Coding ──────────────────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Use Case — Coding & Development");
  sub(s, "Democratizing Software Development", 1.0);
  s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:1.42, w:12.3, h:0.72, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, "In 2020, you needed a developer to build software.\nIn 2025, a business analyst with AI tools can build production applications.",
    0.6, 1.47, 12.1, 0.62, { size:13, italic:true, color:TEAL, align:"center" });
  tbl(s,
    ["Role", "How AI Helps", "Tool Recommendation"],
    [
      ["Software developers", "55% faster; AI handles boilerplate and tests", "Cursor, GitHub Copilot"],
      ["Business analysts", "Build scripts, automate Excel/data tasks", "ChatGPT, Claude"],
      ["Data scientists", "Generate and debug Python/R code", "Cursor, Claude"],
      ["Product managers", "Prototype ideas without developers", "Replit AI, Cursor"],
      ["Finance professionals", "Automate Excel macros, build models", "Claude, ChatGPT"],
      ["Operations", "Build internal tools, automate workflows", "Cursor, Replit"],
    ],
    0.5, 2.25, 12.3, 2.75, [3.2,5.8,3.3],
    { hdrSize:11, rowSize:11 }
  );
  body(s, "What AI coding tools can build (no developer needed):", 0.5, 5.1, 6.5, 0.38, { color:TEAL, bold:true, size:13 });
  bullets(s,
    ["Web scrapers to collect competitor pricing data",
     "Excel automation that processes 10,000 rows in seconds",
     "Internal dashboards connected to your database",
     "Data cleaning and transformation pipelines"],
    0.5, 5.52, 5.9, 1.65, { size:12 }
  );
  s.addShape(pres.shapes.RECTANGLE, { x:6.7, y:5.1, w:6.1, h:2.1, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, "The new core skill for business professionals:", 6.8, 5.18, 5.9, 0.38, { color:TEAL, bold:true, size:12 });
  body(s, "Describe what you want precisely enough that an AI coding tool can execute it — then evaluate the result. This is learnable, and it is now a career differentiator.",
    6.8, 5.6, 5.9, 1.5, { size:12, color:WHITE });
}

// ─── SLIDE 30: Use Case — Business Automation ─────────────────────────────────
{
  let s = ns();
  topBar(s);
  slideTitle(s, "Use Case — Business Automation");
  sub(s, "The Operational Transformation", 1.0);
  tbl(s,
    ["Department", "Automation Opportunity", "Tool", "Time Saved"],
    [
      ["Customer Service", "Tier-1 ticket resolution", "Intercom Fin, Zendesk AI", "40–60% of tickets"],
      ["Sales", "Lead qualification + outreach personalization", "HubSpot AI, Outreach", "5–10 hrs/rep/week"],
      ["HR", "Job description writing, resume screening", "ChatGPT, Claude", "3–5 hrs/hire"],
      ["Finance", "Invoice data extraction, expense categorization", "Claude, GPT-4o", "6–8 hrs/week"],
      ["Marketing", "Content repurposing across channels", "Jasper, Copy.ai", "10–15 hrs/week"],
      ["Legal", "Contract first-pass review and redlining", "Harvey AI, Claude", "50–70% draft time"],
      ["Operations", "SOP creation, process documentation", "Notion AI, Claude", "4–6 hrs/process"],
    ],
    0.5, 1.45, 12.3, 3.6, [2.3,4.5,2.9,2.6],
    { hdrSize:11, rowSize:10 }
  );
  body(s, "The automation stack that works:", 0.5, 5.2, 5, 0.35, { color:TEAL, bold:true, size:13 });
  bullets(s,
    [{text:"LLM (Claude/GPT-4o)", bold:true, color:TEAL},
     "for intelligence — reads, writes, decides",
     {text:"Zapier, Make, or n8n", bold:true, color:TEAL},
     "for connecting your apps",
     {text:"Your existing software", bold:true, color:TEAL},
     "(Salesforce, Gmail, Slack) as the interface",
     {text:"Human oversight", bold:true, color:YELLOW},
     "for exceptions and quality control"],
    0.5, 5.6, 5.9, 2.0, { size:12 }
  );
  s.addShape(pres.shapes.RECTANGLE, { x:6.7, y:5.2, w:6.1, h:2.1, fill:{color:CARD}, line:{color:TEAL,pt:1} });
  body(s, "The question is no longer whether AI automation is real.\nThe question is which department you start with.",
    6.8, 5.45, 5.9, 1.6, { size:14, italic:true, color:WHITE, align:"center", valign:"middle" });
}

// ─── SLIDE 31: Wrap-Up ────────────────────────────────────────────────────────
{
  let s = ns("0A1628");
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:W, h:0.1, fill:{color:TEAL}, line:{color:TEAL} });
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:H-0.1, w:W, h:0.1, fill:{color:TEAL}, line:{color:TEAL} });
  slideTitle(s, "Wrap-Up — The LLM Landscape in 5 Sentences", 0.22, TEAL, 26);
  const sentences = [
    "The Transformer architecture (2017) started everything — every LLM today builds on it.",
    "The ChatGPT moment (Nov 2022) brought AI to a billion people in months.",
    "You have two broad choices: paid/proprietary (easiest, most capable) and free/open-source (private, customizable, scalable).",
    "The real value isn't the LLM — it's the tools built on top that fit into your existing workflows.",
    "Your job: match the right tool to the right task — not be loyal to any one platform.",
  ];
  sentences.forEach((text, i) => {
    s.addShape(pres.shapes.RECTANGLE, { x:0.5, y:1.1 + i*0.82, w:0.45, h:0.45,
      fill:{color:TEAL}, line:{color:TEAL} });
    body(s, String(i+1), 0.5, 1.1 + i*0.82, 0.45, 0.45, { size:14, bold:true, color:"0D1B2A", align:"center" });
    body(s, text, 1.1, 1.13 + i*0.82, 11.7, 0.42, { size:13, color:i===4?TEAL:WHITE, bold:i===4 });
  });
  // Exercises preview
  body(s, "Your 8 Hands-On Exercises This Week:", 0.5, 5.22, 8, 0.38, { color:TEAL, bold:true, size:14 });
  tbl(s,
    ["#", "Tool", "Task"],
    [
      ["1","ChatGPT","Analyze a business document with GPT-4o"],
      ["2","Claude","Long-document risk analysis"],
      ["3","Gemini + NotebookLM","Research paper podcast generation"],
      ["4","Perplexity","Market research vs. Google Search"],
      ["5","GitHub Copilot / Cursor","AI-assisted coding"],
      ["6","Meta AI","Compare free vs. paid LLM output"],
      ["7","Ollama","Run an LLM locally on your computer"],
      ["8","Groq","Blazing-fast API inference"],
    ],
    0.5, 5.65, 12.3, 1.62, [0.4,2.6,9.3],
    { hdrSize:10, rowSize:9 }
  );
}

// ─── Speaker Notes (Professor Scripts, one per slide) ────────────────────────
const NOTES = [

// Slide 1
`Good evening everyone, and welcome to Class 8.

Tonight's topic is one I'm genuinely excited to teach, because I think it's going to change how you see the entire AI landscape — not just the tools you use, but why they exist, who made them, and how to make smart decisions about which one to reach for in any given situation.

We're talking about Large Language Models. LLMs.

Now, I know that term gets thrown around constantly. You've probably used ChatGPT. Maybe you've tried Claude or Gemini. But there's a big difference between using a tool and understanding it. And tonight, we're going to do both.

Here's the journey we're taking together over the next two hours. We're going to start at the very beginning — 2017, a research paper that changed everything. Then we're going to walk through who the major players are today, what tools have been built on top of these models, and — this is the part that matters most for your careers — how you actually choose the right tool for the right business problem.

By the end of tonight, you won't just know what ChatGPT is. You'll know when to use it instead of Claude. You'll know why a company might run an AI model on their own servers instead of paying for an API. You'll understand the ecosystem.

Let's get into it.`,

// Slide 2
`Let me walk you through what we're covering tonight so you know exactly where we're headed.

We have six sections. We'll start with a quick foundation — what is an LLM, in plain terms, no jargon. That's about ten minutes.

Then we go into history. I love this part. The story of how we got from a 2017 Google research paper to a product that had a million users in five days is one of the most dramatic stories in the history of technology. We'll spend about twenty-five minutes there.

After that, we look at the paid, proprietary LLMs — ChatGPT, Claude, Gemini, Perplexity. Twenty minutes. We go platform by platform and I'll show you what each one is actually good at, because they are not the same tool.

Then we flip to the free and open-source side. Llama, Mistral, Ollama, Groq. Another twenty minutes. This is where things get really interesting for businesses that care about privacy or cost at scale.

Then we look at the tools ecosystem — everything built on top of these models. The productivity tools, the coding assistants, the enterprise software, and the no-code automation platforms. That's another twenty-five minutes.

And we close with use cases. Concrete scenarios. How do you actually choose the right LLM for your specific business problem? That's thirty minutes, and it includes a preview of your homework assignment.

So — a full two hours. Let's make every minute count.`,

// Slide 3
`Before we go anywhere, let's make sure we have a solid foundation.

What is a Large Language Model? I want you to understand this in a way you can explain to a colleague who's never heard the term.

Three words: Large. Language. Model. Each one means something specific.

Large means it has been trained on an enormous amount of data — we're talking about a significant fraction of all text ever published on the internet, plus books, scientific papers, code, and more. And it has billions — sometimes hundreds of billions — of internal parameters. Think of parameters as the knobs the model adjusts during training to get better at predicting what comes next.

Language means it operates on text. It reads text in, and it generates text out. Now, modern models can also handle images, audio, and video — but text is still the foundation.

And Model means it's a mathematical system. At its core, what an LLM is doing is predicting the most statistically likely next word given everything that came before it. That's it. It's a very sophisticated autocomplete. The magic is that when you train this on enough data with enough parameters, something extraordinary emerges: the model learns to reason, to write coherently, to follow instructions, to solve problems.

Why does this matter for business? Because tasks that used to require expensive human expertise — drafting contracts, analyzing reports, writing code, answering customer questions — can now be done at scale, in seconds, for fractions of a cent per query.

That's the foundation. Now let's talk about how we got here.`,

// Slide 4
`The story of LLMs starts with a paper.

In 2017, a team at Google Brain published a research paper called "Attention Is All You Need." Eight authors. Forty-three pages. And it changed everything.

The key innovation was something called the Transformer architecture, and specifically a mechanism called self-attention. What self-attention allows a model to do is learn which words in a sentence relate to which other words — no matter how far apart they are. Previous approaches — the RNNs and LSTMs we used before — had to process text sequentially, word by word, and they struggled to maintain context over long distances. The Transformer blew that limitation away.

Every major LLM you interact with today — ChatGPT, Claude, Gemini, Llama — is built on this architecture. Every single one.

Then in 2018, two important things happened almost simultaneously.

Google released BERT. BERT was designed for understanding text — reading it and extracting meaning — not generating it. It's bidirectional, meaning it reads in both directions at once. Google used BERT to massively improve their search results, and they still use descendants of it today.

And OpenAI released GPT-1. The first Generative Pre-trained Transformer. It had 117 million parameters and it could generate coherent paragraphs. Proof of concept that this approach worked.

A year later, GPT-2. One and a half billion parameters — more than ten times bigger. And it was so good at generating convincing text that OpenAI initially refused to release it publicly. They said it was "too dangerous." That's an interesting moment to reflect on — the first time a lab delayed an AI release out of concern about misuse.

That's 2017 to 2019. The spark is lit. Now watch what happens when we pour fuel on it.`,

// Slide 5
`2020 is when things get wild.

OpenAI releases GPT-3. One hundred and seventy-five billion parameters. That's a hundred times bigger than GPT-2. And the jump in capability is not linear — it's exponential. Something happens when you scale a model to this size that nobody fully expected.

The model develops what researchers call emergent abilities. Capabilities that nobody explicitly programmed. GPT-3 could translate languages it wasn't specifically trained to translate. It could do basic arithmetic. It could write code. It learned these things from reading enough text about them.

The other breakthrough in GPT-3 was few-shot learning. You could put three or four examples of a task right in your prompt, and the model would understand the pattern and follow it. This made LLMs practically useful for a huge variety of tasks without any specialized training.

And then in 2021, OpenAI takes GPT-3 and fine-tunes it on code — billions of lines of GitHub repositories — and releases it as Codex. Codex becomes the engine behind GitHub Copilot. For the first time, a professional developer can describe what they want in plain English and get working code back. That's a massive economic disruption for software development.

Also in 2021, DALL-E. OpenAI uses LLM-style training to generate images from text descriptions. The multimodal era begins.

Now, 2022. This is the year of a critical technique called RLHF — Reinforcement Learning from Human Feedback. Instead of just predicting the next token, models start learning to be helpful. Human trainers rate responses. The model learns what "good" looks like from a human perspective. This is the breakthrough that makes LLMs usable by non-experts.

And on November 30th, 2022, OpenAI releases ChatGPT — built on GPT-3.5. One million users in five days. One hundred million in two months. The fastest product adoption in the history of technology. The world changes overnight.`,

// Slide 6
`2023 is the year that everything accelerates at once.

In March, OpenAI releases GPT-4. It passes the bar exam. It scores in the 90th percentile on the SAT. It writes code, analyzes images, reasons through complex problems. Near-human expert performance on a huge range of tasks.

But here's what's interesting — OpenAI is no longer alone.

Anthropic releases Claude. A company founded by ex-OpenAI researchers who left over concerns about safety. Claude is built differently — with a strong emphasis on being harmless and being honest. And critically, it debuts with a 100,000 token context window. That's enormous. You can feed it entire books.

Google releases Bard, which eventually becomes Gemini. They're playing catch-up, but they have an enormous advantage — integration with every Google product billions of people already use.

And then in July 2023, Meta does something that shakes the entire industry. They release the weights for Llama 2. Open source. Free to download. Anyone can run a state-of-the-art LLM on their own servers, for free, with no API calls, no per-token cost, full privacy. The open-source movement explodes.

In September, Mistral releases a 7 billion parameter model that outperforms Llama 2 at 13 billion. Efficiency becomes the new arms race.

2024 brings multimodal everywhere and something new — reasoning models. OpenAI releases o1 and then o3, models that actually think before they answer. They spend extra compute working through a problem step by step. The quality on hard math, science, and logic problems improves dramatically.

And now, in 2025, we're in the era of agents. LLMs that don't just answer questions — they take actions. They browse the web, write files, call APIs, run code. And the cost? What cost twenty dollars per million tokens in 2023 costs ten cents today. The economics are changing as fast as the technology.

That's where we are. Now let's meet the players.`,

// Slide 7
`Alright, now that we have the history, let's look at the landscape as it exists today.

I want you to think about this market in two dimensions. The vertical axis goes from proprietary — meaning the company keeps the model weights secret — down to open source, where the weights are publicly available. The horizontal axis goes from paid to free.

In the top section, the proprietary world, you have the big names. OpenAI with ChatGPT. Anthropic with Claude. Google with Gemini. And then Perplexity, which is a different category — more of a search engine built on top of LLMs.

Down in the open-source world, you have Meta with Llama. Mistral from France. Google's Gemma. Microsoft's Phi. And then platforms like Groq and Ollama that let you run these open models without building your own infrastructure.

Now here's something important I want you to internalize: the line between paid and free is blurring fast. Almost every paid model has a free tier. And almost every open-source model can be accessed for free through one platform or another. The real distinction is about control, privacy, and quality ceiling.

When you're choosing an LLM for a business use case, you're navigating this map. You're asking: How much do I want to pay? How much control do I need over my data? What quality level does this task require? And how much setup am I willing to do?

Keep this map in mind as we go through each player. By the end, you'll know exactly where to land on it for any given situation.`,

// Slide 8
`Let's start with the market leader. OpenAI.

OpenAI was founded in 2015 as a non-profit AI safety research lab. They had a mission to ensure that artificial general intelligence benefits all of humanity. Fast forward a decade, they've taken thirteen billion dollars from Microsoft, they have a consumer product with hundreds of millions of users, and they're valued at over a hundred and fifty billion dollars. Non-profit origins, very for-profit reality.

Their flagship model right now is GPT-4o — and that "o" stands for omni, meaning it handles text, images, audio, and eventually video natively. It's fast, it's capable, and it's the most widely integrated AI model in the world.

They also have the o-series — o3 and o3-mini — which are their reasoning models. These models are different. Before they answer, they essentially think out loud — they work through the problem step by step. This makes them dramatically better at complex math, science, and multi-step reasoning. The trade-off is they're slower and more expensive.

On pricing: the free tier of ChatGPT gives you GPT-4o-mini, which is actually pretty capable. ChatGPT Plus at twenty dollars a month gives you full GPT-4o access, image generation with DALL-E, and access to the GPT store. ChatGPT Pro at two hundred dollars a month unlocks o1 Pro mode with essentially unlimited usage.

For the API — which is how developers integrate OpenAI into their own products — you pay per token. Roughly two-fifty to fifteen dollars per million input tokens depending on the model.

One thing I want to flag on privacy: on the standard ChatGPT plan, your conversations may be used to train future models. If you're putting confidential business information into ChatGPT on a personal account, be aware of that. The Enterprise plan has zero data retention, but that starts at thirty dollars per user per month.

OpenAI's biggest strength isn't just the model — it's the ecosystem. Three million custom GPTs built by users. Integrations with thousands of third-party apps. The widest developer adoption by far.`,

// Slide 9
`Now let's talk about Anthropic and Claude. And this is one I'm particularly interested in, because Anthropic has a fascinating origin story.

In 2021, a group of researchers left OpenAI — including some of the most senior people at the company. Their concern? That the pace of AI development was outrunning our ability to ensure it was safe. So they founded Anthropic with an explicit mission: AI safety research and building AI that is reliably helpful, harmless, and honest.

Their model family is called Claude — currently Claude 4. And there are three tiers within it. Opus is the most powerful, designed for complex analysis and deep reasoning. Sonnet is the balanced one — highly capable but faster and cheaper than Opus. And Haiku is the lightweight, high-speed model for tasks where you need quick responses at scale.

What makes Claude genuinely different from ChatGPT? A few things.

First, the context window. Claude supports up to two hundred thousand tokens. To put that in perspective, that's roughly a hundred and fifty thousand words — you could feed it an entire novel, a full year of emails, or an entire codebase. GPT-4o goes to a hundred and twenty-eight thousand. That gap matters enormously for tasks like contract review, document analysis, or working with large datasets.

Second, instruction following. Claude is exceptionally precise at doing exactly what you ask. If you give it a structured format, a word limit, a specific tone — it follows those instructions more reliably than most other models.

Third, the Projects feature. Claude remembers context across sessions within a project. You can give it background about your company, your writing style, your preferences — and it retains that every time you open a new conversation.

Pricing is similar to OpenAI — twenty dollars a month for Pro, thirty for Team. And like OpenAI, they have an Enterprise tier with zero data retention and full admin controls.

The honest limitation of Claude is its smaller ecosystem. Fewer native integrations, no image generation built in, smaller developer community. But on pure model quality — especially for long documents and precise task execution — Claude is exceptional.`,

// Slide 10
`Google. The company that invented the Transformer architecture and then watched OpenAI build a billion-dollar business with it before they could ship a consumer product. That's one of the great what-ifs of recent technology history.

Their response is Gemini. And Gemini has something that no other model family has: a one million token context window.

Let me put that in perspective. One million tokens is roughly seven hundred and fifty thousand words. You could feed Gemini an entire library of documents. You could give it a year's worth of customer support tickets. You could give it an entire software codebase. The context window isn't just a technical spec — it changes what's possible.

Gemini comes in four flavors. Ultra is the most capable, designed for the hardest tasks. Pro 1.5 is their balanced workhorse. Flash is optimized for speed and cost efficiency — it's extremely fast and cheap to run via API. And Nano runs on-device — it's what powers AI features on Google Pixel phones without any internet connection.

But Gemini's biggest advantage isn't the model itself. It's the integration. If your organization runs on Google Workspace — and a huge percentage of companies do — Gemini is already inside Gmail, Google Docs, Google Sheets, Google Slides, and Google Meet. You're not adopting a new tool. You're activating AI inside the tools your team already uses every day.

And then there's NotebookLM. I want to spend a moment on this because I think it's genuinely one of the most underrated AI products available right now. You upload your documents — PDFs, YouTube videos, Google Docs, websites — and NotebookLM becomes an expert on your specific content. You can ask it questions, get summaries, generate study guides. And it has a feature called Audio Overview that generates a ten-minute podcast — two AI hosts having a real conversation about your documents. It's remarkable.

Pricing follows the same structure: free tier, twenty dollars a month for Google One AI Premium which includes Workspace Gemini features. Enterprise pricing at thirty dollars per user.

Where Gemini falls short is pure text reasoning. On complex multi-step logic problems, it's still a step behind OpenAI and Anthropic on many benchmarks. But for Google Workspace users and for tasks requiring massive context — Gemini is a serious contender.`,

// Slide 11
`Now, Perplexity is an interesting one because it's not quite in the same category as the others. It's not trying to be the most powerful general-purpose LLM. It's doing something more specific — and it does it exceptionally well.

Perplexity is a search engine powered by AI. The core insight behind it is this: every answer should be verifiable. When you ask Perplexity a question, it searches the web in real time, synthesizes what it finds, and gives you a structured answer with cited sources. Every claim traces back to a link you can click.

This solves the single biggest problem with traditional LLMs for research tasks — hallucination. ChatGPT and Claude don't know what happened yesterday. They have a knowledge cutoff. Perplexity doesn't have that problem because it's always searching.

Under the hood, Perplexity uses multiple models depending on the task — GPT-4o, Claude, and their own model called Sonar. You don't really control which one it uses; it selects automatically.

The free tier is genuinely useful — unlimited basic searches, a handful of Pro searches per day. At twenty dollars a month for Pro, you get unlimited deep searches, the ability to upload files, and image generation.

Where Perplexity shines is market research, competitive intelligence, academic research, and fact-checking. If you need to know what a competitor announced last week, what the current market size of an industry is, or what a regulation actually says — Perplexity is the right tool.

Where it doesn't shine is creative tasks, long-form content generation, or conversational depth. It's a research tool, not a writing assistant.

The business case I make for Perplexity is this: use it alongside ChatGPT or Claude, not instead of them. Start your research with Perplexity to get grounded in facts with sources, then bring those insights to Claude or ChatGPT for deeper analysis and content generation.`,

// Slide 12
`Alright, let's pull this all together into something you can actually use.

I want you to look at this comparison table not as a ranking — there is no single winner — but as a decision matrix. Different tools win in different scenarios.

Let me walk you through the key rows.

Context window: Claude wins here at two hundred thousand tokens. Gemini wins if you need a million. ChatGPT is the most limited of the group at a hundred and twenty-eight thousand, though that's still enormous for most tasks.

Starting price: They're all twenty dollars a month for the consumer tier. You're not choosing based on price at this level.

Privacy: This is where they diverge meaningfully. Standard ChatGPT trains on your data. Claude and Gemini do not by default. For the Enterprise tier of any of these, you get zero data retention. If you're handling sensitive business information, this matters.

Now let me give you the quick decision guide that I actually use in practice.

If you need to analyze a very long document — a hundred-page report, a thick contract, a large codebase — start with Claude. Two hundred thousand tokens, exceptional instruction following.

If you need images, voice, or video — go to ChatGPT with GPT-4o. OpenAI still leads on multimodal.

If your organization lives in Google Workspace — Gemini. The integration advantage is enormous.

If you need current information with sources — Perplexity. Don't ask ChatGPT what happened last month when Perplexity can tell you with citations.

If you need to solve a genuinely hard reasoning problem — math, science, logic — OpenAI's o3 is in a different league for that specific use case.

These tools are not competing for the same job. The smart business professional keeps two or three of them in their toolkit and knows which one to reach for in which situation. That's what we're building toward tonight.`,

// Slide 13
`Before we dive into the specific free and open-source models, I need to make sure we're all speaking the same language. Because "free" in the AI world means three very different things, and confusing them will lead you to make bad decisions.

The first meaning is a free API tier. This is where a proprietary model — one that the company keeps locked up — gives you limited usage at no cost. Gemini's free API tier. Claude's free plan. OpenAI's free tier. The model is still owned and controlled by the company. Your data still goes to their servers. You're just not paying in dollars — you're paying with usage limits.

The second meaning is open-weight models. This is fundamentally different. When a company releases open weights, they're making the actual internal parameters of the model — the thing that makes it smart — publicly downloadable. You can take those weights, put them on your own server, and run the model yourself. Meta's Llama, Mistral, Google's Gemma, Microsoft's Phi — these are open-weight models. No API calls. No per-token cost after the initial download. Full control.

And the third meaning is free hosted interfaces. This is the best of both worlds for getting started quickly. Someone else runs the open-weight model on their infrastructure and gives you free access. Ollama lets you run Llama on your own laptop. Groq runs Llama on their custom hardware and gives you a free API. Hugging Face hosts hundreds of models with free tiers.

Now, why does this distinction matter for business? The data privacy column is the critical one. On a free API tier, your data goes to the provider's servers. On a self-hosted open-weight model, your data never leaves your building. If you're in healthcare, finance, legal, or any regulated industry — that's not a minor detail. That's a compliance requirement.

The cost column matters too. A free API tier is free until you hit the limit. At scale — millions of API calls a month — the per-token cost of proprietary models can become substantial. Open-weight models eliminate that completely. The cost becomes infrastructure, which you likely already have.

Keep these three categories in mind as we walk through the specific models.`,

// Slide 14
`Let's talk about Meta AI and Llama 3, because this is arguably the most important development in the AI industry since ChatGPT — and it gets far less attention in the mainstream press.

Meta — Facebook's parent company — made a strategic decision that is genuinely unusual in the technology industry. They decided to give away their most advanced AI models for free. Not as a marketing play. As a deliberate business strategy.

Here's their reasoning: Meta's business is advertising. They don't sell AI. So making their AI infrastructure open-source doesn't hurt their revenue. But it does accelerate adoption, builds goodwill with the developer community, and creates a massive ecosystem of people who know how to work with Llama — which indirectly benefits Meta's core platforms.

The result is Llama 3, which comes in several sizes. The smallest — 1 billion and 3 billion parameters — can run on a phone or a Raspberry Pi. Genuinely. An AI model on a device with no internet connection. The mid-size models, around 8 to 11 billion parameters, run well on a modern laptop or a basic gaming PC. And the 405 billion parameter version — the full-size flagship — approaches GPT-4 quality and requires serious computing infrastructure.

Where can you access Llama 3? Multiple places. Meta.ai is their consumer interface — free, works in the browser. Ollama lets you run it locally on your own machine. Groq hosts it in the cloud and gives you a free API tier with extraordinary speed. Amazon, Microsoft Azure, and Google Cloud all offer it as a hosted option.

The business case for Llama is strongest in two scenarios. First, high volume. If you're making millions of API calls a month, eliminating the per-token cost of proprietary models can save you tens of thousands of dollars. Second, sensitive data. If you're processing employee records, patient information, financial data, or anything that cannot leave your network — running Llama locally means zero data exposure.

The catch is technical complexity. Running your own AI infrastructure requires more setup than signing up for ChatGPT. But the tooling around it — Ollama especially — has gotten remarkably approachable.`,

// Slide 15
`Mistral AI is a company I want you to pay attention to, because they represent something important: the idea that you don't need to be bigger to be better.

Mistral was founded in April 2023 in Paris, France. Three founders — all former researchers from DeepMind and Meta. They raised funding from Andreessen Horowitz, Lightspeed, NVIDIA, and others. And within months of founding, they released their first model.

The headline result of Mistral 7B — their first open-weight release — was this: a 7 billion parameter model that outperformed Meta's Llama 2 at 13 billion parameters on nearly every benchmark. Half the size, better performance. How? By being extremely careful about data quality and training methodology rather than just throwing more compute at the problem.

Then they released Mixtral. And this is where it gets technically interesting. Mixtral uses an architecture called Mixture of Experts, or MoE. Instead of activating all of the model's parameters for every single token, Mixtral has multiple specialized sub-networks — the experts — and for each token, it only activates the two most relevant ones. The result is a model that has 45 billion total parameters but only activates 12 billion at a time. You get the quality of a large model at the cost of a much smaller one.

Their product line today covers several tiers. The open-weight models — Mistral 7B and Mixtral 8x7B — you can download and run for free. Their commercial API models — Mistral Small, Medium, and Large — sit in different price and performance tiers. And Codestral is their code-specialized model.

Pricing for their API is among the most competitive in the industry — often fifty to eighty percent cheaper than equivalent OpenAI models for similar quality.

And one more thing worth mentioning: Mistral is European. They're GDPR-compliant by design. Their data centers are in the EU. For European companies or any organization doing business in Europe with data residency requirements — this is a meaningful advantage that goes beyond just model quality.`,

// Slide 16
`We've covered the big proprietary players and the major open-source players. Now let me introduce you to two models that might be the most underappreciated in the entire landscape — Gemma from Google and Phi-3 from Microsoft.

Both of these are what the industry calls small language models. They're not trying to be the most powerful. They're trying to be the most efficient. And they've succeeded remarkably.

Gemma is Google DeepMind's open-weight model family. The same team that builds Gemini. Think of it as Gemini's open-source cousin. The 2 billion and 7 billion parameter versions can run on a laptop or even a phone. Gemma 2, their second generation, at 9 and 27 billion parameters, is genuinely competitive with models twice its size.

The business case for Gemma is developers who want to embed Google-quality AI into their own applications without paying API costs, and organizations that need AI to run on-device — no internet required, no latency, no data leaving the machine.

Now, Microsoft Phi-3. This one has a fascinating research story behind it. The team at Microsoft Research asked a question: what if instead of training on all the text on the internet — including a lot of garbage — we train exclusively on very high-quality, textbook-like content? And then they trained a tiny model on that.

The result is astonishing. Phi-3 Mini has 3.8 billion parameters — small enough to run on a smartphone — and it beats GPT-3.5 on many reasoning benchmarks. Phi-3 Medium at 14 billion parameters approaches GPT-4 performance on some tasks.

Why does this matter practically? If Phi-3 can give you ninety percent of GPT-4's quality at one percent of the compute cost, running locally with zero latency and zero API fees — the math is compelling for a wide range of business applications. Think about a field sales representative with no reliable internet connection who needs AI assistance on their tablet. Or a manufacturing quality control system that can't afford cloud API latency. These are the scenarios where small, efficient, on-device models win decisively.`,

// Slide 17
`We've talked about the models themselves. Now let's talk about two platforms that change how you run those models — Groq and Ollama. These are different from each other in almost every way, but they both address a real limitation: access.

Let's start with Groq. Groq — spelled G-R-O-Q — is a hardware and cloud company. They built a completely custom chip called the Language Processing Unit, or LPU, specifically designed for one thing: running LLM inference as fast as physically possible.

How fast? Standard cloud APIs from OpenAI or Anthropic typically deliver somewhere between 50 and 80 tokens per second — that's roughly the speed you read text. Groq delivers 250 to 800 tokens per second. That's 5 to 10 times faster. Responses appear essentially instantaneously, even for long outputs.

They run open-source models on this hardware — Llama 3, Mixtral, Gemma — and provide free API access with rate limits, and paid tiers that start at extremely competitive prices.

When does speed matter? More than you might think. Voice interfaces cannot have a two-second delay before responding. Real-time translation cannot lag. Customer service chatbots that feel slow get abandoned. High-throughput data processing pipelines where you're running a thousand API calls in a batch — speed becomes cost. Groq is the answer for all of these.

Now, Ollama is the opposite in almost every dimension. Groq is cloud-hosted, fast, and public. Ollama is local, runs on your own machine, and your data never goes anywhere.

Ollama is a piece of software you install on your Mac, Windows, or Linux machine. You then pull any of a hundred-plus supported open-weight models — Llama, Mistral, Phi, Gemma, and more — and run them locally. The command is simple: "ollama run llama3.2" and you're chatting with an AI model with zero internet required.

For businesses, Ollama's value proposition is absolute data privacy. Your HR documents, your legal contracts, your financial records — you can run sophisticated AI analysis on sensitive data without that data ever touching an external server.

Together, Groq and Ollama represent the two extremes of the open-source deployment story — one optimized for speed in the cloud, one optimized for privacy on your own hardware.`,

// Slide 18
`Let's bring this section together the same way we did for paid models — with a decision framework you can actually use.

Look at this table. Six platforms, six criteria. Let me walk you through the rows that matter most for business decisions.

Privacy is the most important for regulated industries. Anything you run locally — Llama, Mistral, Gemma, Phi-3 through Ollama — is completely private. Your data never leaves your machine. Groq is in the middle — it's cloud-hosted, but it's Groq's cloud rather than OpenAI's or Anthropic's, and their terms are more permissive. If you have strict data residency or compliance requirements, locally-hosted is the only acceptable answer.

Cost is straightforward. The open-weight models are free — you pay for the hardware to run them, which you likely already have. Groq has a generous free tier and then extremely competitive paid tiers.

Setup effort is where the proprietary free tiers win decisively. Signing up for Claude.ai takes ninety seconds. Installing Ollama and pulling a model takes about ten minutes but requires a bit of comfort with the terminal.

Let me give you a practical framework for choosing. If your primary concern is data privacy — run locally with Ollama. If your primary concern is speed for a high-throughput application — use Groq. If your primary concern is cost at scale — open-weight models self-hosted. If you just want to try something out quickly at no cost — the free tiers of Claude or Gemini are the path of least resistance.

And here's the strategic takeaway for this entire section: the existence of high-quality open-source models is putting meaningful downward pressure on the pricing of proprietary models. OpenAI and Anthropic cannot charge whatever they want because Meta and Mistral are giving away competitive alternatives. This is good for every business that uses AI.`,

// Slide 19
`Alright, we've spent a lot of time talking about the models themselves — who makes them, how they compare, what they cost. Now I want to zoom out and show you something more important for your day-to-day work: the ecosystem of tools that sits on top of those models.

Because here's the thing — most of you are never going to interact with a raw LLM directly. You're not going to call the OpenAI API and write JSON. You're going to use applications. And those applications are all built on top of these same underlying models.

Think about it like plumbing. The LLM is the water supply. The API is the pipes. And the apps you actually use — ChatGPT, Notion AI, GitHub Copilot, Salesforce Einstein — those are the faucets. You don't care about the plumbing. You care about the faucet working.

Now, there are three categories of tools in this ecosystem.

The first is native apps — tools built by the same company that makes the model. ChatGPT is OpenAI's native app. Claude.ai is Anthropic's. Gemini is Google's. These have the tightest integration and usually the best access to new features.

The second category is third-party tools — independent companies that built products on top of LLM APIs. Cursor is a coding IDE built on Claude and GPT-4o. Jasper is a marketing tool built on GPT-4o. Perplexity is a search engine built on multiple models. These companies add specialized functionality on top of the foundation.

The third category is enterprise integrations — large software vendors who have embedded AI into products you already use. Microsoft put Copilot into Word, Excel, Teams, and Outlook. Salesforce built Einstein. Zendesk built AI triage. You may already be paying for these capabilities right now.

Here's the insight I want you to hold onto: the model itself is increasingly becoming a commodity. The real business value — and where companies are winning and losing — is in the application layer built on top. The LLM is the engine. What matters is the car.`,

// Slide 20
`OpenAI started as a research lab. Today they are one of the most strategically important platform companies in the world. And the reason is that they didn't just build a model — they built an ecosystem.

Let me walk you through the pieces.

At the consumer level, you have ChatGPT — which is already familiar to most of you. But inside ChatGPT, there's DALL-E 3 for generating images from text. There's Sora for generating videos from text. There's a voice mode that lets you have a spoken conversation with the AI. And there's a canvas feature for collaborative document editing with AI. That's a lot for twenty dollars a month.

For developers, OpenAI has the Assistants API — which lets you build AI agents with persistent memory, file access, and tool use. They have an embeddings API for building search and recommendation systems. And the GPT Store, where anyone — without writing a single line of code — can create a custom version of ChatGPT trained on their own materials. There are now three million of these custom GPTs, ranging from customer service bots to specialized legal assistants to cooking advisors.

And at the enterprise level, they have Azure OpenAI Service — which is OpenAI's models running inside Microsoft's cloud infrastructure with enterprise security, compliance certifications, and service agreements. That's the version major corporations use when they need GPT-4o but can't put their data on OpenAI's consumer servers.

The strategic play here is clear: OpenAI is trying to be the operating system for AI. Not just a model provider. A platform. The GPT Store is their App Store equivalent. The API is their developer platform. And the enterprise layer is how they win the Fortune 500.

Whether they succeed is still an open question — the competition is intense. But they have a head start measured in years.`,

// Slide 21
`Let's look at two very different philosophies for building an AI ecosystem.

Anthropic's approach is what I'd call quality over quantity. They have fewer products than OpenAI, but each one is exceptionally well executed. The flagship is Claude.ai — their chat interface with a free tier and a Pro tier. What makes it different from ChatGPT in practice is the Projects feature. Claude remembers things across sessions within a project. You can give it context about your company, your role, your preferences, and it carries that context every time you open a new conversation. For ongoing business work — writing in a consistent style, working on a long project — that's genuinely useful.

They also have Artifacts, which is their equivalent of Canvas — you can generate and edit documents, code, and diagrams live inside the conversation. And their API is where things get interesting for developers: Claude Sonnet is currently the most popular model used in Cursor and is a major part of the GitHub Copilot backend for complex tasks. So even when you're using a third-party coding tool, there's a good chance you're actually running Claude underneath.

Google's approach is the opposite — integration everywhere. Their philosophy is that you shouldn't have to adopt a new tool. AI should be inside the tools you already use.

And they have the scale to execute on that. Gemini is embedded in Gmail, Google Docs, Google Sheets, Google Slides, and Google Meet. If your organization runs on Google Workspace — and a huge proportion of businesses do — Gemini is already available to you without any new software purchase.

But the product I genuinely think is underrated is NotebookLM. You upload documents — PDFs, YouTube videos, articles, Google Docs — and it becomes an expert on your specific content. Ask it questions. Get summaries and study guides. And then it does something I've never seen any other tool do: it generates an Audio Overview — two AI hosts having a ten-minute podcast conversation about your documents. You can hand that to a new employee and they'll understand a complex report in their commute. That is a genuinely new capability.`,

// Slide 22
`There is a category of tools that I think will affect more business professionals more immediately than any other part of today's lecture. I'm talking about productivity and writing tools — the applications that put AI directly into the content creation workflow.

Every major writing platform is adding AI right now. And the market for these tools is exploding. Let me walk you through the major players.

Notion AI sits inside your notes and wikis. If your team uses Notion — which many startups and tech companies do — AI is already in there. Ask it to summarize a long meeting note. Ask it to turn bullet points into a formal document. Ask it to translate your internal jargon into customer-facing language.

Jasper is specifically designed for marketing copy. It knows how to write ad copy, blog posts, email campaigns. And critically, you can train it on your brand voice — upload examples of your existing content, and it learns to write the way your company writes. That consistency is hard to achieve manually at scale.

Copy.ai is similar but with a stronger focus on sales and marketing sequences. Email drip campaigns, cold outreach, LinkedIn messages.

Grammarly you likely already use for grammar and spell-check. Their AI layer now does much more — it rewrites sentences for tone, adjusts formality, suggests alternatives for clarity. It's a writing coach embedded in every text field you use.

Gamma is worth knowing for presentations specifically. You write a prompt — "create a ten-slide deck on our Q2 marketing strategy" — and it generates the full deck with layout, imagery, and content. Not perfect, but a genuinely useful starting point.

And I should mention HeyGen, because you'll hear about it in this course. It generates AI avatar videos from a text script. You write what you want a presenter to say, it generates a video of that person saying it. We're using it to produce some of the video content for this course.

The key pattern across all of these tools: they take a foundational LLM, add domain-specific templates and brand training, and wrap it in an interface designed for a specific use case. You're not paying for the model. You're paying for the packaging.`,

// Slide 23
`This is the category that has probably created the most economic disruption of any AI application so far. Developer tools.

GitHub Copilot launched in 2021 and now has nearly two million paying subscribers. GitHub's own internal study found that developers using Copilot code fifty-five percent faster. That is a massive productivity gain. If you have a team of twenty engineers and they're all fifty-five percent more productive, that's effectively eleven engineers worth of work for free.

Let me walk you through the landscape.

GitHub Copilot is the original. It sits inside VS Code, JetBrains, and other IDEs. As you type, it suggests completions — not just the next word, but entire functions, entire classes. It reads the code around you and predicts what you're trying to build. Under the hood, it uses Claude and GPT-4o.

Cursor is what I'd call the next generation. It's a full IDE — you can switch to it from VS Code — and it goes much further than autocomplete. You can open a chat with the AI, show it your entire codebase, and ask it to understand a bug, explain a function, or make a change across multiple files. It's less like an autocomplete tool and more like a pair programmer who has read every line of your code.

Windsurf is similar to Cursor in the agentic space — it can make multi-file edits autonomously, not just suggest what you should type.

Replit AI is interesting for non-developers specifically because you don't install anything. You build and run code in the browser. The AI helps you write it. You can prototype an application in a browser tab.

Now, here's what I want business students to take away from this slide. These tools are not just for software engineers. A data analyst can use Cursor or Claude to write Python scripts to process data. A finance professional can use ChatGPT to generate Excel macros. A product manager can prototype an application without ever hiring a developer. The barrier between "business person" and "person who can build software" is lower than it has ever been in history.`,

// Slide 24
`I want to make sure you understand something important about AI adoption in the enterprise context. For most large organizations, the question is not "should we adopt AI?" The question is "are we actually using the AI we already paid for?"

Because here's the reality: if your company uses Microsoft 365, you're almost certainly already paying for Microsoft Copilot, or it's being offered to you. If you use Salesforce, Einstein AI is already in your CRM. If you use Zendesk for customer service, their AI features are already part of your subscription. The AI is there. The adoption isn't.

Let me go through the categories.

In CRM and sales, Salesforce Einstein Copilot can summarize sales calls, draft follow-up emails, and give probability scores to deals in your pipeline. HubSpot AI can write entire email campaigns and landing pages from a brief. Outreach personalizes cold outreach at scale — instead of writing the same cold email a hundred times, the AI generates a personalized version for each prospect based on their company and role.

In customer service, Intercom's Fin is the most interesting product right now. It's a fully autonomous AI agent that can resolve over fifty percent of support tickets without a human ever seeing them. Questions it can't answer, it escalates. The economics are remarkable — if your support team handles ten thousand tickets a month and Fin resolves five thousand of them, you've cut your support workload in half.

In productivity, Microsoft Copilot 365 is the most ambitious play. It's embedded in every Office application — Word, Excel, PowerPoint, Teams, Outlook. Summarize a long email thread. Get meeting notes from Teams. Ask Excel to analyze your data and build a chart. Ask PowerPoint to create a slide deck from a document.

The hidden opportunity here is that most companies pay thirty dollars per user per month for Microsoft Copilot 365 and use twenty percent of its features. Learning to maximize these tools you already have is immediate ROI with zero additional spend. That's the first place I'd start in any organization.`,

// Slide 25
`The last category of tools I want to cover is one that I think creates the most opportunity for people in this room specifically. Because it requires no coding. No technical background. No developers. And yet it lets you build workflows that are genuinely sophisticated.

I'm talking about no-code AI automation platforms.

Let me give you the basic concept. In the past, if you wanted to connect multiple applications together and have them do something automatically, you needed to write code. API integrations, webhooks, data transformations — all developer work. Zapier changed that in 2012 by making it visual. But Zapier was automating simple, rule-based actions — if this happens, do that.

What no-code AI automation adds is intelligence to those workflows. It's not just "if new email arrives, forward it." It's "if new email arrives, read it with an LLM, determine if it's a complaint or a request, draft an appropriate response, update the CRM record, and send a Slack message to the account manager." That's a workflow that would have required a developer to build in 2022. Today a business analyst can build it in a tool like n8n or Make in an afternoon.

Let me highlight n8n specifically because I think it's exceptional. It's open-source — you can run it on your own server, which means your data never leaves your building. It integrates with four hundred apps. And its AI Agent node lets you give the LLM tools — the ability to search the internet, query a database, send an email — and then let it decide on its own which tools to use to complete a task. That's getting close to what we call an AI agent.

The other platforms — Zapier, Make, Flowise, Dify, Relevance AI, Voiceflow — each have their sweet spots. Zapier is the most beginner-friendly with six thousand app integrations. Flowise is open-source and specifically designed for building chatbots and retrieval-augmented generation systems. Voiceflow is for conversational AI across channels — web, mobile, WhatsApp, phone.

The critical thing to understand about all of these: most of them let you swap the underlying LLM without rebuilding the workflow. You start with GPT-4o for quality. You find out the cost is too high at volume, so you switch to Mistral. Your client has a privacy requirement, so you switch to Ollama running locally. The workflow stays the same. The intelligence changes. That flexibility is genuinely powerful for business.`,

// Slide 26
`Alright, we've covered a lot of ground tonight. History. Models. Tools. Now let's bring it all together into something actionable — a decision framework you can actually use tomorrow morning.

The most common mistake I see people make with AI tools is loyalty. They pick one tool — usually ChatGPT because it was first — and they use it for everything. That's like using a hammer for every job because it was the first tool you found. Hammers are great for nails. They're terrible for screws.

So here is how I think about choosing the right tool.

Start with the task. What are you actually trying to do? If you're writing content — blog posts, emails, marketing copy — ChatGPT or Claude are your workhorses. If you need current information with citations — something that happened last month, market data, a competitor's recent announcement — Perplexity is the right answer. If you're working with a very long document — a hundred-page contract, an entire research report — Claude with its two-hundred-thousand token context window is where you go. If you're coding, or want to build something — Cursor or GitHub Copilot. If your entire team lives in Google Workspace — Gemini is already there, already integrated, already paid for.

Then check your budget. The good news is that the free tiers today are genuinely capable. Gemini free, Claude free, and Meta AI on the open-source side — none of these would have seemed possible two years ago. If you're paying twenty dollars a month for any one tool, you're in the premium tier.

And then — this is the one most people skip — think about your data. If what you're putting into the AI is public information or non-sensitive, any cloud LLM is fine. If it's confidential business strategy, customer data, employee records, anything regulated — you need to either use an enterprise plan with zero data retention, or you need to run locally with Ollama. This is not optional in regulated industries. It's a compliance requirement.

The framework is simple: task first, budget second, data sensitivity third. In that order.`,

// Slide 27
`Let me give you a concrete look at what AI-powered content creation actually looks like in practice, because I think this is where most business professionals will see the fastest and most tangible return.

McKinsey published research in 2024 showing that marketing teams using AI produce three to five times more content at sixty percent lower cost. That is not a marginal improvement. That is a structural change in how marketing gets done.

What can AI actually create? Pretty much everything. Blog posts of a thousand to three thousand words — high quality, with editing — in minutes instead of hours. Social media posts for every platform, calibrated to the right tone and format for each. Email campaigns and drip sequences that would take a copywriter days to write. Ad copy for Google, Meta, LinkedIn. Product descriptions for e-commerce catalogs with hundreds of SKUs. Video scripts. Press releases. Presentation decks.

Now I want to be honest about what "high quality" means here. AI doesn't produce publish-ready content on a first draft the way a great human writer might. What it produces is an excellent starting point that requires editing — but that editing is much faster than starting from scratch. The comparison isn't AI versus a great writer. The comparison is AI-plus-human-editor versus a human writer working alone. The AI-plus-human combination wins on speed and volume almost every time.

The workflow I recommend looks like this. Start by briefing the AI thoroughly — give it your brand voice, your audience, your goal, any key points you want to hit. Then ask it for three variants, not one. You want options. Then a human editor refines the best version, fact-checks it, adds anything proprietary or anecdotal that the AI can't know, and publishes.

I'll give you a real example. A five-person marketing team at a mid-size B2B company used Claude and Jasper to increase their blog output from four posts a month to twenty posts a month without adding a single headcount. Organic traffic grew a hundred and forty percent in six months. That's not a hypothetical. That's what's happening in the market right now.`,

// Slide 28
`Research is one of the highest-leverage applications of AI for business professionals, and I think it's dramatically underused relative to its potential.

Let me paint you a picture of the traditional research process. You need to understand a market. You start searching. You find twenty articles. You read them all. You take notes. You synthesize. You write. You go back to check your citations. You realize you missed something. You search again. From start to a solid written analysis — four to eight hours. For a complex topic, more.

The AI-augmented version looks like this. You open Perplexity and do a targeted search on the market, the competitors, the trends. It searches the web in real time and gives you a structured synthesis with clickable citations in minutes. Then you take the most important documents — PDFs, reports, articles — and put them into NotebookLM. It reads everything, connects the dots, and you can ask it questions across all fifty sources simultaneously. Then you bring your notes to Claude for deeper analysis — Claude can hold your entire research brief in context and write a structured output. Total elapsed time: thirty to sixty minutes.

Let me break down what each tool does best in a research workflow.

Perplexity is where you start for anything current. Market size, competitor activity, regulatory changes, recent news — Perplexity searches the web in real time and cites its sources. You can verify every claim.

NotebookLM is your research analyst. You feed it your documents — and it can handle up to fifty sources — and then you interrogate them. Ask it to find everything the documents say about pricing strategy. Ask it to build a timeline of industry events. And then generate the audio podcast — two AI hosts discussing your documents — which is remarkable for consuming dense material on a commute.

Claude is where you go for synthesis and analysis of long, complex documents. Feed it an entire annual report. Ask for the five biggest risk factors. Feed it a legal contract and ask where the liability language is unusual. Two hundred thousand tokens means you almost never hit a limit.`,

// Slide 29
`I want to make a bold claim and then back it up with evidence.

The claim is this: in 2025, a business analyst with no prior coding experience can build a working, production application using AI tools. Not a toy prototype. A real application.

That would have been false in 2022. It is true today.

Here's how.

For professional software developers, the change is already well-documented. GitHub Copilot has nearly two million subscribers. Developers using it code fifty-five percent faster on average. That's the equivalent of getting a hundred developers and having them suddenly become a hundred and fifty-five. The economic impact on software teams is enormous.

But I want to focus on the people in this room who are not developers. Because I think the opportunity is actually bigger for you.

If you are a business analyst and you need to process ten thousand rows of customer data — segment it, clean it, transform it, summarize it — you can ask Claude or ChatGPT to write you a Python script to do it. You don't need to understand Python. You need to be able to describe what you want clearly, and then evaluate whether the output is correct.

If you are a finance professional and you need a complex Excel macro to automate a monthly reporting process, you can ask an AI to write the VBA code. Paste it into Excel. Test it. If something's wrong, paste the error message back to the AI. Iterate until it works.

If you are a product manager and you want to test whether an idea is viable before spending engineering resources — Replit AI runs in the browser. You can describe an application in plain English, have the AI build it, and demo it to stakeholders — all without a single developer involved.

The new core skill for business professionals is what I call AI-mediated technical communication. The ability to describe what you want in precise enough terms that an AI coding tool can execute it, and then evaluate the result. That is a learnable skill. And it is now a career differentiator.`,

// Slide 30
`Let's talk about what happens when you connect an LLM to the rest of your business software. Because the individual productivity gains we've been discussing are meaningful. But the organizational gains from automation are an order of magnitude larger.

Let me give you three concrete workflows that organizations are running right now.

First: customer email management. A customer sends an email to support. An AI — running through something like n8n or Zapier — reads the email, categorizes it as a billing issue, a technical question, or a complaint. It searches the knowledge base for relevant answers. It drafts a response. It routes the email to the right team. It logs everything in the CRM. A human reviews the draft, edits if needed, and approves it. What used to take fifteen minutes of human attention per email — reading, researching, composing, routing — is now three minutes of review.

Second: sales operations. A sales call ends. The call recording goes to an AI. The AI transcribes it, extracts the action items, identifies what the prospect said their key concerns are, and updates the Salesforce deal record automatically. It drafts a follow-up email for the sales rep to review and send. The rep opens their laptop and sees their entire call already documented and their email already written.

Third: HR onboarding. A new employee starts. An AI onboarding bot is available twenty-four hours a day to answer questions — where's my benefits portal, who do I call for IT, what's the vacation policy. It generates a personalized thirty-day learning plan based on the employee's role. It tracks completion of required training and flags gaps to the manager automatically.

Now look at the ROI table on this slide. Customer service: forty to sixty percent of tier-one tickets resolved without human involvement. Sales: five to ten hours per rep per week freed up. HR: three to five hours saved per hire. Finance: six to eight hours a week on invoice processing and expense categorization. Marketing: ten to fifteen hours a week on content repurposing.

These are not speculative numbers. These are what organizations deploying these tools are reporting. The question is no longer whether AI automation is real. The question is which department you're going to start with.`,

// Slide 31
`Two hours. Thirty-one slides. Let me give you the five sentences that capture everything.

The Transformer architecture, published by Google researchers in 2017, is the foundation of every LLM you will interact with. The ChatGPT moment in November 2022 changed the trajectory of the technology industry overnight — a hundred million users in two months. You have two broad categories of models to choose from: paid and proprietary, which are the easiest to start with and the most capable; and free and open-source, which give you privacy, customizability, and cost control at scale. The real value creation is not in the models themselves — it's in the tools built on top of them that fit into the workflows of your specific job and industry. And your most important skill as a business professional in this environment is not knowing how to build AI — it's knowing how to choose the right tool for the right task.

Now, for this week, you have eight exercises. I want you to actually do them. Not read about them — do them. Run Ollama on your laptop. Use Groq's free API. Put a document into NotebookLM. Compare Meta AI to ChatGPT on the same prompt. These tools are free to access. The exercises are designed to take thirty to sixty minutes each. By the time you've done all eight, you'll have hands-on experience with every major category of LLM we discussed tonight.

And then next class — I want you to come in ready to talk about what surprised you. What was better than you expected. What was worse. Where you think the gaps are. Because that critical perspective — not just enthusiasm, but evaluated judgment — is what separates someone who uses AI from someone who understands it.

The question you should be asking yourself from now on is never "should I use AI for this?" That question is over. The answer is almost always yes.

The question is: which AI, for this task, at this cost, with this data sensitivity level?

That's the question of a professional. And that's what we've been building toward all night.

See you next class.`,

];

// Apply notes to each slide
NOTES.forEach((note, i) => {
  if (pres.slides[i]) pres.slides[i].addNotes(note);
});

// ─── Write file ───────────────────────────────────────────────────────────────
const OUT = "/Users/app210006/Documents/GoogleDrive/Classes/AA_Courses/UTDallas/Spring 2026/BUAN 6v99.sw2 - Special Topics in Business Analytics - Generative AI for Business - Online/Class 8/Week08_LLM_Slides.pptx";

pres.writeFile({ fileName: OUT })
  .then(() => console.log("✅  Saved:", OUT))
  .catch(err => { console.error("❌  Error:", err); process.exit(1); });
