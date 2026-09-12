import { useState } from 'react'
import { motion } from 'framer-motion'
import { Loader2, Play, TrendingUp } from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'
import { runSimulation } from '@/lib/api'
import type { SimulateResponse } from '@/types'

interface SliderRowProps {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  format?: (v: number) => string
}

function SliderRow({ label, value, onChange, min, max, step, format }: SliderRowProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <span className="font-mono text-xs">{format ? format(value) : value}</span>
      </div>
      <Slider value={[value]} onValueChange={(v) => onChange(v as number)} min={min} max={max} step={step} />
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="py-4">
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="font-mono text-xl font-semibold">{value}</div>
      </CardContent>
    </Card>
  )
}

const THRESHOLD_SWEEP = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

export function SimulationDashboard() {
  const [nPatients, setNPatients] = useState(2000)
  const [nCentres, setNCentres] = useState(20)
  const [bandwidth, setBandwidth] = useState(5)
  const [nReviewers, setNReviewers] = useState(10)
  const [threshold, setThreshold] = useState(0.95)

  const [result, setResult] = useState<SimulateResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [sweepData, setSweepData] = useState<{ threshold: number; reviewer_utilization: number; sensitivity_lost: number }[] | null>(null)
  const [sweeping, setSweeping] = useState(false)

  const baseParams = {
    n_patients: nPatients,
    n_centres: nCentres,
    nurses_per_centre: 2,
    bandwidth_mbps: bandwidth,
    n_reviewers: nReviewers,
    auto_clear_threshold: threshold,
  }

  const handleRun = async () => {
    setRunning(true)
    try {
      const res = await runSimulation(baseParams)
      setResult(res)
    } finally {
      setRunning(false)
    }
  }

  const handleSweep = async () => {
    setSweeping(true)
    try {
      const rows = await Promise.all(
        THRESHOLD_SWEEP.map(async (t) => {
          const res = await runSimulation({ ...baseParams, auto_clear_threshold: t })
          return {
            threshold: t,
            reviewer_utilization: res.reviewer_utilization ?? 0,
            sensitivity_lost: res.sensitivity_lost ?? 0,
          }
        })
      )
      setSweepData(rows.reverse())
    } finally {
      setSweeping(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">District screening simulation</h1>
        <p className="text-sm text-muted-foreground">
          A discrete-event model of a district screening programme. Triage routing -- auto-clearing
          high-confidence grade-0 cases -- is the lever that trades reviewer load against missed
          referable cases; the sweep below is grounded in the real trained model's own confidence
          distribution on its validation set, not an assumed error rate.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">Parameters</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <SliderRow label="Patients simulated" value={nPatients} onChange={setNPatients} min={200} max={5000} step={100} />
            <SliderRow label="Primary health centres" value={nCentres} onChange={setNCentres} min={1} max={50} step={1} />
            <SliderRow
              label="Upload bandwidth"
              value={bandwidth}
              onChange={setBandwidth}
              min={2}
              max={20}
              step={1}
              format={(v) => `${v} Mbps`}
            />
            <SliderRow label="Ophthalmologist reviewers" value={nReviewers} onChange={setNReviewers} min={1} max={30} step={1} />
            <SliderRow
              label="Auto-clear threshold"
              value={threshold}
              onChange={setThreshold}
              min={0.3}
              max={0.99}
              step={0.01}
              format={(v) => `P(grade=0) >= ${v.toFixed(2)}`}
            />

            <div className="flex gap-2 pt-2">
              <Button onClick={handleRun} disabled={running} className="flex-1">
                {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                Run
              </Button>
              <Button onClick={handleSweep} disabled={sweeping} variant="secondary" className="flex-1">
                {sweeping ? <Loader2 className="h-4 w-4 animate-spin" /> : <TrendingUp className="h-4 w-4" />}
                Sweep threshold
              </Button>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-6">
          {result && (
            <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-3 gap-3">
              <StatCard label="Auto-cleared" value={`${result.auto_cleared}/${result.patients_out}`} />
              <StatCard label="Reviewed" value={`${result.reviewed}/${result.patients_out}`} />
              <StatCard
                label="Sensitivity lost"
                value={result.sensitivity_lost != null ? `${(result.sensitivity_lost * 100).toFixed(2)}%` : 'n/a'}
              />
              <StatCard label="Turnaround p50" value={`${Math.round(result.turnaround_p50_s)}s`} />
              <StatCard label="Turnaround p95" value={`${Math.round(result.turnaround_p95_s)}s`} />
              <StatCard
                label="Reviewer utilization"
                value={result.reviewer_utilization != null ? `${(result.reviewer_utilization * 100).toFixed(1)}%` : 'n/a'}
              />
            </motion.div>
          )}

          {sweepData && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm text-muted-foreground">
                  Reviewer load vs sensitivity lost, by auto-clear threshold
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={sweepData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                    <XAxis dataKey="threshold" stroke="var(--color-muted-foreground)" fontSize={11} />
                    <YAxis stroke="var(--color-muted-foreground)" fontSize={11} />
                    <Tooltip contentStyle={{ background: 'var(--color-card)', border: '1px solid var(--color-border)', borderRadius: 8 }} />
                    <Line type="monotone" dataKey="reviewer_utilization" stroke="#3B82F6" name="Reviewer utilization" strokeWidth={2} dot={false} />
                    <Line type="monotone" dataKey="sensitivity_lost" stroke="#EF4444" name="Sensitivity lost" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {!result && !sweepData && (
            <Card className="flex h-64 items-center justify-center border-dashed">
              <p className="text-sm text-muted-foreground">Run a simulation to see results here.</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
