# srgan-rust Pricing

**Neural network image upscaling — as a service.**
Start for free. Scale when you need to.

---

## Plans

### Free
**$0 / month — no credit card required**

Everything you need to evaluate and prototype.

- 10 images per day
- Up to 4 MP input resolution (e.g. 2000×2000)
- All three built-in models (natural, anime, waifu2x)
- Auto model detection
- Watermark on output images
- Synchronous API only
- 5 requests / minute rate limit
- Community support (GitHub Issues)

[Get your free API key →](#)

---

### Pro
**$19 / month**

For indie developers and small teams shipping production features.

- **1,000 images per day**
- Up to **20 MP input resolution** (e.g. 5000×4000)
- All built-in models — no restrictions
- **No output watermark**
- **Async job queue** — submit and poll, don't block your server
- **Webhooks** — HMAC-signed POST callbacks when jobs complete, with automatic retry
- **S3 output** — results written directly to your bucket, presigned URLs returned
- 60 requests / minute rate limit
- Priority queue placement (ahead of Free tier)
- 99.5% uptime SLA
- Email support (response within 1 business day)

[Start Pro trial — 14 days free →](#)

---

### Enterprise
**Custom pricing**

For teams with high volume, compliance requirements, or specialized needs.

- **Unlimited images** — no daily cap
- **Unlimited input resolution**
- **Custom models** — bring your own `.rsr`/`.bin` model, or commission a domain-specific one (medical imaging, satellite, product photography…)
- No watermark
- Async queue with **highest priority** — Enterprise jobs process before Pro and Free
- Dedicated worker pool (no noisy-neighbour contention)
- **On-premises deployment** — run in your own infrastructure, air-gapped if needed
- **SLA 99.9% uptime** with financial penalties
- Custom rate limits
- Prometheus metrics endpoint + Grafana dashboard
- Dedicated Slack channel and named account manager
- Net-30 invoicing available

[Contact sales →](#)

---

## Feature comparison

| Feature | Free | Pro | Enterprise |
|---------|:----:|:---:|:----------:|
| Images per day | 10 | 1,000 | Unlimited |
| Max input resolution | 4 MP | 20 MP | Unlimited |
| Output watermark | Yes | No | No |
| Built-in models | All | All | All |
| Custom models | — | — | Yes |
| Auto model detection | Yes | Yes | Yes |
| Synchronous API | Yes | Yes | Yes |
| Async job queue | — | Yes | Yes |
| Job priority | Lowest | Medium | Highest |
| Webhooks | — | Yes | Yes |
| S3 output | — | Yes | Yes |
| Rate limit | 5 req/min | 60 req/min | Custom |
| On-premises | — | — | Yes |
| Uptime SLA | — | 99.5% | 99.9% |
| Support | Community | Email | Dedicated |
| Invoicing | — | Credit card | Net-30 available |

---

## FAQ

**Can I use the self-hosted version for free?**
Yes. srgan-rust is MIT-licensed open source. You can run it on your own infrastructure at no cost. The hosted service above is for teams who prefer not to manage the infrastructure themselves.

**What counts as an "image"?**
One POST to `/api/v1/upscale` or one item processed in a batch job counts as one image. Polling `/api/v1/job/:id` does not count.

**What happens when I hit my daily limit?**
The API returns HTTP 429 with a `Retry-After` header indicating when your quota resets (midnight UTC). Your existing in-flight jobs will complete.

**How are input megapixels counted?**
Width × height of the input image in pixels, divided by 1,000,000. A 2000×2000 image is 4 MP.

**Can I upgrade mid-month?**
Yes. Upgrades are prorated to the day. You'll have access to the new tier immediately.

**Do you offer annual billing?**
Yes — annual Pro is $190/year (saves two months). Contact us for Enterprise annual pricing.

**Is there a free trial for Pro?**
Yes — 14 days, no credit card required. Full Pro quota and features, no watermark.

**What payment methods do you accept?**
All major credit cards via Stripe. Enterprise customers can request invoicing (Net-30).

**Can I run the API on my own servers?**
Yes. See [docs/DEPLOYMENT.md](DEPLOYMENT.md) for Docker, systemd, and Kubernetes deployment guides.

---

## Volume pricing

Need more than 1,000 images/day but prefer a pay-per-use model? We offer volume packs:

| Pack | Images | Price | Per image |
|------|--------|-------|-----------|
| Starter | 5,000 | $49 | $0.0098 |
| Growth | 25,000 | $199 | $0.0080 |
| Scale | 100,000 | $599 | $0.0060 |
| Enterprise | Custom | Custom | Custom |

Packs do not expire. They stack on top of any active subscription.

[Purchase a volume pack →](#)

---

*Prices in USD. All plans billed monthly unless stated. We reserve the right to change pricing with 30 days notice to existing subscribers.*
