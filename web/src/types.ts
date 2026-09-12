export type QualityVerdict = 'GRADEABLE' | 'ENHANCEABLE' | 'REJECT'

export interface PointingGameScore {
  hits: number
  total: number
  score: number | null
}

export interface NeovascularisationResult {
  flag: boolean
  rule_based: boolean
  tortuosity: number
  branching_density: number
}

export interface ScreeningResponse {
  session_id: string
  quality_verdict: QualityVerdict
  quality_reject_reason: string | null
  severity_grade: number | null
  referable: boolean | null
  raw_referable_score: number | null
  calibrated_confidence: number | null
  lesion_counts: Record<string, number>
  od_center: [number, number] | null
  od_confidence: number | null
  fovea_center: [number, number] | null
  fovea_confidence: number | null
  neovascularisation: NeovascularisationResult | null
  gradcam_native_size: [number, number] | null
  pointing_game_scores: Record<string, PointingGameScore> | null
  images: {
    original: string | null
    lesion_overlay: string | null
    gradcam: string | null
  }
}

export interface SimulateRequest {
  n_patients: number
  n_centres: number
  nurses_per_centre: number
  bandwidth_mbps: number
  n_reviewers: number
  auto_clear_threshold: number
  seed?: number
}

export interface SimulateResponse {
  patients_in: number
  patients_out: number
  auto_cleared: number
  reviewed: number
  sensitivity_lost: number | null
  turnaround_p50_s: number
  turnaround_p95_s: number
  reviewer_utilization: number | null
}

export const SEVERITY_COLORS: Record<number, string> = {
  0: '#10B981',
  1: '#84CC16',
  2: '#F59E0B',
  3: '#F97316',
  4: '#EF4444',
}

export const SEVERITY_LABELS: Record<number, string> = {
  0: 'No DR',
  1: 'Mild NPDR',
  2: 'Moderate NPDR',
  3: 'Severe NPDR',
  4: 'Proliferative DR',
}

export const LESION_LABELS: Record<string, string> = {
  ma: 'Microaneurysms',
  he: 'Haemorrhages',
  ex: 'Hard exudates',
  se: 'Soft exudates',
}
