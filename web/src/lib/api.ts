import axios from 'axios'
import type { ScreeningResponse, SimulateRequest, SimulateResponse } from '@/types'

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

const client = axios.create({ baseURL: API_BASE })

export async function screenImage(file: File, degradeSeverity = 0): Promise<ScreeningResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('degrade_severity', String(degradeSeverity))
  const { data } = await client.post<ScreeningResponse>('/screen', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export function reportUrl(sessionId: string): string {
  return `${API_BASE}/report/${sessionId}`
}

export async function runSimulation(req: SimulateRequest): Promise<SimulateResponse> {
  const { data } = await client.post<SimulateResponse>('/simulate', req)
  return data
}

export function dataUri(base64: string | null): string | null {
  return base64 ? `data:image/png;base64,${base64}` : null
}
