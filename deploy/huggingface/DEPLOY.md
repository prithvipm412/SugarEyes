# Deploying to Hugging Face Spaces

Requires a free Hugging Face account -- this is a manual, one-time push from
your machine; nothing here runs automatically.

1. Create a new Space at huggingface.co/new-space: SDK = **Docker**, hardware
   = free CPU tier (this API is CPU-only, see AGENTS.md).
2. Clone the Space's own git repo it gives you a URL for, separately from
   this project's GitHub repo.
3. Copy into that Space repo: the root `Dockerfile`, `requirements.txt`,
   `src/`, `api/`, `configs/`, `models/manifest.json`, and this file's
   sibling `README.md` (the one with the Space's YAML frontmatter --
   Spaces require it at the repo root, which is why it lives here rather
   than replacing the project's own root README).
4. Get trained checkpoints onto the Space. The free tier has no persistent
   volume by default -- either enable persistent storage (paid) and copy
   checkpoints in once, or add a startup step that downloads them from
   wherever you're hosting the trained weights (e.g. a private HF Model
   repo, which is free and a natural fit here).
5. `git add -A && git commit -m "deploy" && git push` to the Space's remote.
   The Space builds the Dockerfile and starts serving automatically.
6. Check `https://<your-space>.hf.space/health` -- `models_loaded: true`
   confirms the checkpoints actually resolved.

For the demo to work with no camera/upload friction, see the fixture images
under `deploy/fixtures/` -- precomputed results for 6 showcase images, so a
live demo never waits on a cold pipeline run for its first impression.
