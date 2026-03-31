# Week 08 — The World of Large Language Models
# Batch 3 of 5: Free & Open-Source LLMs (Slides 13–18)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 13: What Does "Free" Really Mean?

### Three Very Different Things

**1. Free API Tier** (Still proprietary, just free up to a limit)
- Examples: Gemini API free tier, Claude.ai free plan, OpenAI free tier
- The company still owns the model
- Usage limits apply — rate limits, message caps
- Your data may still be used for training

**2. Open-Weight Models** (The model weights are public — you can download them)
- Examples: Meta Llama 3, Mistral 7B, Google Gemma
- Download the model and run it yourself — on your computer or cloud
- No API calls, no per-token cost after download
- Full control: fine-tune it, modify it, deploy it privately
- Requires technical setup and computing resources

**3. Free Hosted Interfaces** (Open-weight models run on someone else's server)
- Examples: Ollama (local), Groq (cloud), Hugging Face Spaces
- You get the free open model but without the setup hassle
- Fastest path to running Llama, Mistral, etc.

**Why this matters for business:**

| Concern | Free API Tier | Open-Weight | Free Hosted |
|---------|---------------|-------------|-------------|
| Data Privacy | ⚠️ Shared with provider | ✅ 100% private | ✅ Mostly private |
| Cost at Scale | Paid beyond limits | Infra cost only | Free or very cheap |
| Customization | ❌ Limited | ✅ Full fine-tuning | Limited |
| Setup Effort | None | High | Low |

---

## Slide 14: Meta Llama 3 — Open Source Champion

### The People's LLM

**Company:** Meta AI (Facebook's parent company)
**Released:** Llama 3.1 (July 2024), Llama 3.2/3.3 (late 2024)
**License:** Open weights — free to download, use, and modify

**Model variants:**

| Model | Parameters | Best For | Can Run On |
|-------|-----------|----------|-----------|
| Llama 3.2 1B / 3B | 1–3 billion | Mobile, edge devices | Phones, Raspberry Pi |
| Llama 3.2 11B / 90B | 11–90 billion | General tasks, vision | Gaming PC, Mac M3 |
| Llama 3.1 405B | 405 billion | Near-GPT-4 quality | Data center / cloud |

**Where to access Llama 3:**

| Platform | Cost | Privacy | Ease of Use |
|----------|------|---------|-------------|
| Meta.ai | Free | Meta sees it | Very easy |
| Ollama (local) | Free | 100% private | Easy |
| Groq | Free tier | Groq sees it | Easy (API) |
| Amazon Bedrock | Pay per use | Your AWS account | Medium |
| Hugging Face | Free / paid | Shared | Medium |

**Business case for Llama:**
- Zero per-token cost at scale (huge savings for high-volume apps)
- Full data privacy — sensitive business data never leaves your servers
- Can be fine-tuned on your company's data and terminology

---

## Slide 15: Mistral AI — The European Challenger

### Quality Over Size

**Company:** Mistral AI (founded 2023, Paris, France)
**Backed by:** Andreessen Horowitz, Lightspeed, NVIDIA
**Philosophy:** Efficient, powerful models that punch above their weight

**Why Mistral matters:**
> Mistral 7B outperformed Llama 2 13B on every benchmark — at half the size

**Model family:**

| Model | Parameters | Type | Best For |
|-------|-----------|------|----------|
| Mistral 7B | 7B | Open-weight | Fast, lightweight tasks |
| Mixtral 8x7B | 45B active (MoE) | Open-weight | High quality at lower cost |
| Mistral Small | — | API | Cost-efficient API usage |
| Mistral Large | — | API | Complex reasoning, code |
| Codestral | — | API | Code generation (all languages) |

**MoE — Mixture of Experts (Mixtral):**
> Instead of using all 45B parameters for every token, Mixtral activates only the 2 most relevant "expert" sub-networks → GPT-3.5 quality at a fraction of the compute cost

**Pricing:**
- Open-weight models: **Free** (download and run yourself)
- Mistral API: Free tier available; paid tiers from ~$0.002/1K tokens
- La Plateforme (enterprise): Custom pricing

**European advantage:** GDPR-compliant by design; data stored in EU — important for European businesses and regulated industries

---

## Slide 16: Google Gemma & Microsoft Phi-3

### Small Models, Big Impact

---

### Google Gemma

**Released:** February 2024 | **Type:** Open weights
**Built by:** Google DeepMind — same team as Gemini

| Model | Parameters | Special Feature |
|-------|-----------|-----------------|
| Gemma 2B | 2 billion | Runs on phones and laptops |
| Gemma 7B | 7 billion | Strong general performance |
| Gemma 2 (9B/27B) | 9–27B | State-of-the-art for its size |
| CodeGemma | 7B | Optimized for code |

**Best for:** Developers who want Google-quality AI embedded in their own apps, on-device AI (no internet required)

---

### Microsoft Phi-3

**Released:** April 2024 | **Type:** Open weights (MIT license)
**Key insight:** Train on high-quality "textbook-like" data rather than raw internet text

| Model | Parameters | Context | Remarkable fact |
|-------|-----------|---------|-----------------|
| Phi-3 Mini | 3.8B | 128K | Fits on a phone, beats GPT-3.5 |
| Phi-3 Small | 7B | 128K | Better than Mixtral 8x7B on reasoning |
| Phi-3 Medium | 14B | 128K | Approaches GPT-4 on some benchmarks |

**Why small models matter for business:**
- Run on-device → **no internet connection needed** (field workers, secure environments)
- **Zero latency** — response in milliseconds
- **No per-call cost** — deploy once, use forever
- **Compliance-friendly** — data never leaves the device

---

## Slide 17: Groq & Ollama — Speed and Privacy

### Supercharging Open-Source Models

---

### Groq — Inference at the Speed of Thought

**What it is:** A cloud platform that runs open-source models at extraordinary speed using custom LPU (Language Processing Unit) chips

**Speed comparison:**
| Platform | Tokens/second | Llama 3.1 70B |
|----------|--------------|---------------|
| OpenAI API | ~60 t/s | (GPT-4o) |
| Standard cloud GPU | ~40 t/s | |
| **Groq** | **~250–800 t/s** | ✅ |

**Models available:** Llama 3.1/3.3, Mixtral, Gemma, Whisper (audio)
**Pricing:** Free tier (rate-limited), paid from $0.05–$0.79 per million tokens
**Best for:** Real-time apps, voice interfaces, high-throughput pipelines

---

### Ollama — Your Private AI on Your Laptop

**What it is:** A free tool to run LLMs locally on your Mac, Windows, or Linux machine

**How simple it is:**
```
Install Ollama → ollama pull llama3.2 → ollama run llama3.2
```

**Supported models:** Llama 3, Mistral, Phi-3, Gemma, CodeLlama, Qwen, and 100+ more
**Hardware needed:** Mac M1/M2/M3 or GPU-equipped PC (8GB+ RAM minimum)
**Cost:** 100% free — no API keys, no internet required after download

**Business use cases:**
- **Sensitive data analysis** — HR files, legal documents, financials — stays on your machine
- **Offline environments** — manufacturing floors, field operations, classified settings
- **Prototype fast** — test AI features locally before committing to cloud costs

---

## Slide 18: Free & Open-Source LLMs — Business Comparison

### The Free Tier Decision Matrix

| | **Llama 3 (Meta)** | **Mistral/Mixtral** | **Gemma (Google)** | **Phi-3 (Microsoft)** | **Groq** | **Ollama** |
|---|---|---|---|---|---|---|
| **Access Method** | Download / API | Download / API | Download / API | Download / API | Cloud API | Local install |
| **Best For** | General tasks | Efficient reasoning | On-device AI | Small + smart | Speed-critical | Privacy-first |
| **Privacy Level** | ✅ If self-hosted | ✅ If self-hosted | ✅ If self-hosted | ✅ On-device | ⚠️ Groq cloud | ✅ 100% local |
| **Cost** | Free weights | Free weights | Free weights | Free weights | Free tier | 100% free |
| **Setup Effort** | Medium | Medium | Medium | Medium | Low (API) | Low |
| **Min. Hardware** | 8GB RAM | 8GB RAM | 4GB RAM | 4GB RAM | None | 8GB RAM |
| **Fine-tunable** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |

**When to choose open-source over paid:**

✅ **Go open-source when:**
- Handling sensitive/confidential business data
- High volume (millions of API calls/month)
- Need to customize/fine-tune on your domain
- Budget is a primary constraint

💳 **Stick with paid when:**
- Need the absolute best quality available
- Speed-to-production matters more than cost
- Team lacks technical AI infrastructure expertise
- Multimodal (vision, voice) is required

---
