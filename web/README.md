# SugarEyes web UI

Vite + React + TypeScript + Tailwind v4 + shadcn/ui (Base UI primitives) +
Framer Motion + Recharts. Design system (dark theme only, deliberately no
light mode) is defined in `src/index.css` -- see the repo root's `AGENTS.md`
for the exact colour tokens and rationale.

## Setup

```bash
npm install
npm run dev          # expects the API at http://127.0.0.1:8000 by default;
                      # override with VITE_API_BASE=http://host:port
```

The API (`api/main.py` at the repo root) must be running separately --
`source ../.venv/bin/activate && uvicorn api.main:app --reload` from the
repo root.

## Structure

- `src/components/UploadScreen.tsx` -- drag-and-drop upload, plus the
  "simulate field capture" degradation control (see AGENTS.md: this project
  has no physical fundus camera, so this is the honest, deliberate stand-in)
- `src/components/ResultView.tsx` -- quality-gate reject panel, severity
  grade, calibrated confidence, Grad-CAM (opacity slider + side-by-side
  toggle), lesion evidence overlay, and the explainability/pointing-game panel
- `src/components/SimulationDashboard.tsx` -- district screening simulation
  controls, live stat cards, and the auto-clear-threshold trade-off chart
- `src/lib/api.ts` -- typed client for `/screen`, `/report/{id}`, `/simulate`
- `src/components/ui/` -- shadcn/ui components (Base UI-backed in this
  shadcn version, not classic Radix -- `Slider`'s `onValueChange` passes a
  plain `number` for a single-thumb slider, not a `[number]` array; `Button`
  has no `asChild` prop, use the exported `buttonVariants()` class helper
  on a plain element instead when you need Button styling on a non-button tag)

## Build

```bash
npm run build   # tsc -b && vite build -- fails on any TypeScript error
```
