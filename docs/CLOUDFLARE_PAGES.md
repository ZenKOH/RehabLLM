# RehabLLM public research site — Cloudflare Pages deployment

This is a **static, source-linked research prototype**, not RehabLLM model inference and not a medical decision-support service.

The existing PyTorch code remains unchanged. All public web files live under `site/`:

- `site/index.html` — accessible, responsive public research page and curated evidence notes;
- `site/assets/styles.css` — design and responsive layout;
- `site/assets/app.js` — client-side evidence search/filter (no network requests);
- `site/_headers` — security and privacy-related response headers;
- `site/favicon.svg`, `site/robots.txt`, `site/404.html` — supporting static assets.

**No Neon project, Clerk account, R2 bucket, Resend API key, inference endpoint, billing account, or Cloudflare Worker is needed for v1.** Avoid giving the site access to patient information or API secrets. The site does not accept form submissions or send questions to a model.

## Test locally

From the repository root:

```sh
python scripts/check_public_site.py
python -m http.server 8787 --directory site
```

Open `http://localhost:8787`. This is a local preview, not a public URL. Test a narrow and wide viewport and the evidence search/filter controls.

## Deploy through Cloudflare Pages with GitHub

1. In Cloudflare's dashboard open **Workers & Pages → Create application → Pages → Connect to Git**.
2. Authorise access to the GitHub repository **`ZenKOH/RehabLLM`** (grant access to this repository rather than every repository where possible).
3. Select **no framework preset** (plain static HTML).
4. Configure:
   - **Root directory:** repository root (leave blank)
   - **Build command:** `exit 0` (or leave blank)
   - **Build output directory:** `site`
   - **Production branch:** `main` **after the pull request has been reviewed and merged**.
   - **Environment variables:** none.
5. Select **Save and Deploy**. Once Cloudflare reports success, use the actual `*.pages.dev` URL provided by Cloudflare. No public hostname is predetermined in this repository.

For **review before merging**, connect the repository and enable Cloudflare Pages preview deployments for pull requests / non-production branches. Cloudflare normally creates preview URLs after Git integration is configured; make sure branch previews are enabled. Do not call a preview a production launch.

Official deployment documentation: https://developers.cloudflare.com/pages/framework-guides/deploy-anything/ and https://developers.cloudflare.com/pages/get-started/git-integration/

**Important:** Cloudflare Pages may publish the `main` branch before this PR merges. Since the existing `main` branch does not contain `site/`, a first production build with output directory `site` can fail. To avoid this, either merge the PR before starting production deployment or configure a temporary preview/production branch pointing at this feature branch and switch to `main` only after review.

## Scope and safeguards

- Static code and content only. There is no model checkpoint in the repository and no public chatbot.
- Model specifications and v0.3 results are labelled as **documented**, not independently reproduced.
- Evidence cards link to the original WHO, NICE and PMC pages and carry interpretation limits. This is not exhaustive evidence synthesis.
- The site sets a restrictive Content Security Policy and uses no third-party scripts, analytics or trackers.
- Any introduction of accounts, paid plans, uploads, model inference, analytics, healthcare data or external APIs must trigger a separate security/privacy review.
- Clinical claims and evidence should be reviewed periodically, especially after guidelines change. The initial notes were checked on **9 November 2026**.

## Later steps (not part of this deployment)

1. Independently reproduce or evaluate trained checkpoint behaviour before creating an inference demo.
2. Prototype an evidence catalogue with structured metadata and source versioning; **Neon** could be used when persistent storage becomes necessary.
3. Consider R2 for large, properly licensed research assets. Add authentication and email only if users actually need accounts or notifications.
4. Conduct a safety, privacy, reliability and regulatory assessment before offering patient-related functionality.
