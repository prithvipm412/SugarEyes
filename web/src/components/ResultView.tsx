import { useState } from 'react'
import { motion } from 'framer-motion'
import { AlertCircle, Download, Eye, RotateCcw } from 'lucide-react'
import { Button, buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Slider } from '@/components/ui/slider'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { dataUri, reportUrl } from '@/lib/api'
import { LESION_LABELS, SEVERITY_COLORS, SEVERITY_LABELS, type ScreeningResponse } from '@/types'

interface Props {
  result: ScreeningResponse
  onReset: () => void
}

export function ResultView({ result, onReset }: Props) {
  const [gradcamOpacity, setGradcamOpacity] = useState(0.6)

  if (result.quality_verdict === 'REJECT') {
    return (
      <div className="mx-auto max-w-xl space-y-4 text-center">
        <Card className="border-destructive/50">
          <CardContent className="space-y-3 py-10">
            <AlertCircle className="mx-auto h-10 w-10 text-destructive" />
            <h2 className="text-lg font-semibold">Image rejected -- recapture required</h2>
            <p className="text-sm text-muted-foreground">
              Reason: <span className="font-medium text-foreground">{result.quality_reject_reason ?? 'unspecified'}</span>
            </p>
            <p className="text-xs text-muted-foreground">
              This image did not pass the automated quality gate and was not graded.
            </p>
          </CardContent>
        </Card>
        <Button variant="outline" onClick={onReset}>
          <RotateCcw className="h-4 w-4" /> Try another image
        </Button>
      </div>
    )
  }

  const grade = result.severity_grade ?? 0
  const gradeColor = SEVERITY_COLORS[grade]
  const confidencePct = result.calibrated_confidence != null ? Math.round(result.calibrated_confidence * 100) : null
  const original = dataUri(result.images.original)
  const lesionOverlay = dataUri(result.images.lesion_overlay)
  const gradcam = dataUri(result.images.gradcam)

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Screening result</h1>
        <div className="flex gap-2">
          <a
            href={reportUrl(result.session_id)}
            target="_blank"
            rel="noreferrer"
            className={buttonVariants({ variant: 'outline', size: 'sm' })}
          >
            <Download className="h-4 w-4" /> PDF report
          </a>
          <Button variant="ghost" size="sm" onClick={onReset}>
            <RotateCcw className="h-4 w-4" /> New image
          </Button>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr]">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">Fundus image &amp; evidence</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="sidebyside">
              <TabsList>
                <TabsTrigger value="sidebyside">Side by side</TabsTrigger>
                <TabsTrigger value="overlay">Grad-CAM overlay</TabsTrigger>
                <TabsTrigger value="lesions">Lesion evidence</TabsTrigger>
              </TabsList>

              <TabsContent value="sidebyside" className="grid grid-cols-2 gap-3">
                {original && <img src={original} alt="original" className="rounded-lg" />}
                {gradcam && <img src={gradcam} alt="grad-cam" className="rounded-lg" />}
              </TabsContent>

              <TabsContent value="overlay" className="space-y-3">
                <div className="relative overflow-hidden rounded-lg">
                  {original && <img src={original} alt="original" className="w-full" />}
                  {gradcam && (
                    <img
                      src={gradcam}
                      alt="grad-cam overlay"
                      className="absolute inset-0 w-full transition-opacity"
                      style={{ opacity: gradcamOpacity }}
                    />
                  )}
                </div>
                <div className="flex items-center gap-3">
                  <span className="w-24 shrink-0 font-mono text-xs text-muted-foreground">
                    opacity {gradcamOpacity.toFixed(2)}
                  </span>
                  <Slider value={[gradcamOpacity]} onValueChange={(v) => setGradcamOpacity(v as number)} min={0} max={1} step={0.05} />
                </div>
              </TabsContent>

              <TabsContent value="lesions">
                {lesionOverlay && <img src={lesionOverlay} alt="lesion overlay" className="w-full rounded-lg" />}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardContent className="flex items-center gap-4 pt-6">
              <div
                className="flex h-16 w-16 shrink-0 items-center justify-center rounded-full font-mono text-2xl font-bold text-white"
                style={{ background: gradeColor }}
              >
                {grade}
              </div>
              <div>
                <div className="font-semibold">{SEVERITY_LABELS[grade]}</div>
                <div className="text-xs text-muted-foreground">ICDR scale, grade {grade} of 4</div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm text-muted-foreground">Calibrated confidence (referable DR)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="font-mono text-2xl font-bold">{confidencePct ?? '--'}%</div>
              <div className="h-3 overflow-hidden rounded-full bg-secondary">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: gradeColor }}
                  initial={{ width: 0 }}
                  animate={{ width: `${confidencePct ?? 0}%` }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                />
              </div>
            </CardContent>
          </Card>

          <Card className={result.referable ? 'border-severity-4/50 bg-severity-4/10' : 'border-severity-0/50 bg-severity-0/10'}>
            <CardContent className="flex items-center justify-between py-4">
              <span className="text-sm font-semibold">
                {result.referable ? 'Refer for ophthalmology follow-up' : 'No referral indicated'}
              </span>
              <Badge variant={result.referable ? 'destructive' : 'secondary'}>{confidencePct ?? '--'}%</Badge>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">Lesion counts</CardTitle>
          </CardHeader>
          <CardContent>
            <table className="w-full text-sm">
              <tbody>
                {Object.entries(LESION_LABELS).map(([cls, label]) => (
                  <tr key={cls} className="border-b border-border last:border-0">
                    <td className="py-2 text-muted-foreground">{label}</td>
                    <td className="py-2 text-right font-mono">{result.lesion_counts[cls] ?? 0}</td>
                  </tr>
                ))}
                {result.lesion_counts.ma_candidates != null && (
                  <tr>
                    <td className="py-2 text-muted-foreground">MA candidates (stage-1, pre-classification)</td>
                    <td className="py-2 text-right font-mono">{result.lesion_counts.ma_candidates}</td>
                  </tr>
                )}
              </tbody>
            </table>
            {result.neovascularisation && (
              <>
                <Separator className="my-3" />
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-muted-foreground">Neovascularisation flag:</span>
                  <span className="font-medium">{result.neovascularisation.flag ? 'POSITIVE' : 'negative'}</span>
                  <Badge variant="outline" className="text-[10px]">
                    RULE-BASED, NOT LEARNED
                  </Badge>
                </div>
              </>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm text-muted-foreground">
              <Eye className="h-4 w-4" /> Explainability evidence
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Pointing-game score: fraction of each lesion class whose location falls inside the
              Grad-CAM's top-10% activation.
              {result.gradcam_native_size && (
                <>
                  {' '}
                  Native CAM resolution is only {result.gradcam_native_size[0]}&times;{result.gradcam_native_size[1]}{' '}
                  -- it cannot reliably localise individual microaneurysms; treat MA scores as a known
                  resolution limit, not a model failure.
                </>
              )}
            </p>
            {result.pointing_game_scores && (
              <table className="w-full text-sm">
                <tbody>
                  {Object.entries(result.pointing_game_scores).map(([cls, score]) => (
                    <tr key={cls} className="border-b border-border last:border-0">
                      <td className="py-2 text-muted-foreground">{LESION_LABELS[cls] ?? cls}</td>
                      <td className="py-2 text-right font-mono">
                        {score.score != null ? `${Math.round(score.score * 100)}%` : 'n/a'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </CardContent>
        </Card>
      </div>
    </motion.div>
  )
}
