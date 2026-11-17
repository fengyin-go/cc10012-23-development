// 趋势预测训练引擎：纯函数，不碰存储。
// 负责样本评估（样本量 / 历史窗 / 特征缺失）、按预测步长训练并生成预测值、训练批次同源校验。
import { shiftDay, type OrderFeatureRow } from './order-features'

// 样本门槛：有效样本至少 30 条、历史窗至少 28 天，且都要覆盖 2 倍预测步长。
export const MIN_SAMPLE_COUNT = 30
export const MIN_HISTORY_DAYS = 28
// 特征空值占比超过 20% 视为特征缺失。
export const FEATURE_NULL_TOLERANCE = 0.2
// 训练窗口：用最近 28 天的有效样本拟合。
const TRAIN_WINDOW_DAYS = 28

export type SampleAssessment = {
  sufficient: boolean
  sampleCount: number
  historyDays: number
  requiredSamples: number
  requiredDays: number
  missingFeatures: string[]
  basis: string
}

export type PredictionPoint = {
  日期: string
  预测值: number
  下限: number
  上限: number
}

function round1(value: number): number {
  return Math.round(value * 10) / 10
}

function validRows(rows: OrderFeatureRow[]): OrderFeatureRow[] {
  return rows
    .filter((row) => typeof row.订单量 === 'number' && Number.isFinite(row.订单量))
    .slice()
    .sort((a, b) => a.日期.localeCompare(b.日期))
}

function spanDays(rows: OrderFeatureRow[]): number {
  if (rows.length === 0) {
    return 0
  }
  const from = new Date(`${rows[0].日期}T00:00:00Z`).getTime()
  const to = new Date(`${rows[rows.length - 1].日期}T00:00:00Z`).getTime()
  return Math.round((to - from) / 86400000) + 1
}

// 样本评估：样本量、历史窗、特征缺失三项都达标才算充足；样本依据原样保留在结果业务数据上。
export function assessSamples(
  rows: OrderFeatureRow[],
  featureFields: string[],
  datasetFields: string[],
  targetField: string,
  step: number,
): SampleAssessment {
  const valid = validRows(rows)
  const sampleCount = valid.length
  const historyDays = spanDays(valid)
  const requiredSamples = Math.max(MIN_SAMPLE_COUNT, step * 2)
  const requiredDays = Math.max(MIN_HISTORY_DAYS, step * 2)

  const missingFeatures: string[] = []
  if (!datasetFields.includes(targetField)) {
    missingFeatures.push(`目标字段「${targetField}」未登记`)
  } else if (sampleCount === 0) {
    missingFeatures.push(`目标字段「${targetField}」无有效值`)
  }
  for (const field of featureFields) {
    if (!datasetFields.includes(field)) {
      missingFeatures.push(`「${field}」未登记`)
      continue
    }
    const nullCount = rows.filter((row) => {
      const value = (row as unknown as Record<string, number | string | null>)[field]
      return value === null || value === undefined
    }).length
    if (rows.length > 0 && nullCount / rows.length > FEATURE_NULL_TOLERANCE) {
      missingFeatures.push(`「${field}」空值过多`)
    }
  }

  const enoughSamples = sampleCount >= requiredSamples
  const enoughDays = historyDays >= requiredDays
  const parts: string[] = []
  parts.push(
    enoughSamples
      ? `有效样本${sampleCount}条≥门槛${requiredSamples}条`
      : `有效样本${sampleCount}条<门槛${requiredSamples}条`,
  )
  parts.push(
    enoughDays
      ? `历史窗${historyDays}天≥门槛${requiredDays}天`
      : `历史窗${historyDays}天<门槛${requiredDays}天`,
  )
  const sufficient = enoughSamples && enoughDays && missingFeatures.length === 0
  const basis = sufficient
    ? `样本充足：${parts.join('，')}`
    : missingFeatures.length > 0
      ? `特征缺失：${missingFeatures.join('、')}；${parts.join('，')}`
      : `样本不足：${parts.join('，')}`

  return { sufficient, sampleCount, historyDays, requiredSamples, requiredDays, missingFeatures, basis }
}

function zValue(confidence: string): number {
  const match = confidence.match(/(\d+(?:\.\d+)?)/)
  const level = match ? Number(match[1]) : 95
  if (level >= 99) return 2.576
  if (level >= 95) return 1.96
  if (level >= 90) return 1.645
  return 1.28
}

// 训练并生成预测值：最近 28 天有效样本做线性趋势拟合，促销特征按历史占比折算期望 uplift，
// 按预测步长逐日外推，置信区间 = 残差标准差 × 置信系数。
export function trainPredict(
  rows: OrderFeatureRow[],
  step: number,
  confidence: string,
  featureFields: string[],
): PredictionPoint[] {
  const valid = validRows(rows)
  const window = valid.slice(-TRAIN_WINDOW_DAYS)
  const n = window.length
  if (n < 2 || step < 1) {
    return []
  }
  const xs = window.map((_, index) => index)
  const ys = window.map((row) => Number(row.订单量))
  const meanX = xs.reduce((sum, x) => sum + x, 0) / n
  const meanY = ys.reduce((sum, y) => sum + y, 0) / n
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < n; i += 1) {
    sxy += (xs[i] - meanX) * (ys[i] - meanY)
    sxx += (xs[i] - meanX) * (xs[i] - meanX)
  }
  const slope = sxx === 0 ? 0 : sxy / sxx
  const intercept = meanY - slope * meanX
  const residualStd = Math.sqrt(
    ys.reduce((sum, y, i) => sum + (y - (intercept + slope * xs[i])) ** 2, 0) / n,
  )

  let uplift = 0
  if (featureFields.includes('促销标记')) {
    const promo = window.filter((row) => row.促销标记 === 1)
    const normal = window.filter((row) => row.促销标记 === 0)
    if (promo.length > 0 && normal.length > 0) {
      const promoMean = promo.reduce((sum, row) => sum + Number(row.订单量), 0) / promo.length
      const normalMean = normal.reduce((sum, row) => sum + Number(row.订单量), 0) / normal.length
      uplift = ((promoMean - normalMean) * promo.length) / n
    }
  }

  const half = zValue(confidence) * residualStd
  const lastDay = window[n - 1].日期
  const points: PredictionPoint[] = []
  for (let t = 1; t <= step; t += 1) {
    const value = intercept + slope * (n - 1 + t) + uplift
    points.push({
      日期: shiftDay(lastDay, t),
      预测值: round1(value),
      下限: round1(Math.max(0, value - half)),
      上限: round1(value + half),
    })
  }
  return points
}

export function predictInterval(points: PredictionPoint[]): string {
  if (points.length === 0) {
    return ''
  }
  return `${points[0].日期}~${points[points.length - 1].日期}`
}

// 训练批次号内嵌模型编号与数据集版本：批次和数据集版本同源可见，
// 报告引用前用 batchMatchesDataset 校验，防止引用训练范围外的预测。
export function makeBatchCode(modelCode: string, datasetCode: string, seq: number): string {
  return `BATCH-${modelCode}-${datasetCode}-${String(seq).padStart(3, '0')}`
}

export function batchMatchesDataset(batch: string, modelCode: string, datasetCode: string): boolean {
  return batch.startsWith('BATCH-') && batch.includes(modelCode) && batch.includes(datasetCode)
}
