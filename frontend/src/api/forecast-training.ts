// 训练执行编排：读取历史订单特征和训练数据集，按预测步长训练并生成预测值，
// 产出「预测模型结果」业务数据（forecastrun）。
// 关键约束：
//   1. 样本不足 / 特征缺失 → 阻断发布，保留上一份可用模型，产出带不足标记的草稿；
//   2. 训练批次内嵌数据集版本，同源可见，报告不会引用训练范围外的预测；
//   3. 竞态重试按（模型编号 + 数据集版本）幂等，只生成一份结果业务数据；
//   4. 中断 / 异常 → 已失败 + 异常原因，允许重试；
//   5. 报告只读取「已生效」且批次同源的结果，不合格草稿永远不会被当成已生效模型。
import { listRows, saveRows } from '@/data/local-store'
import { orderFeaturesOf } from '@/data/order-features'
import {
  assessSamples,
  batchMatchesDataset,
  makeBatchCode,
  predictInterval,
  trainPredict,
  type PredictionPoint,
} from '@/data/forecast-engine'
import type { ActionResult, EntryRow } from '@/data/types'

const RESULT_KEY = 'forecastrun'
const MODEL_KEY = 'forecast'
const DATASET_KEY = 'datastore'

// 非终态：同模型同数据集版本的重试会原地更新，而不是再插一行。
const NON_FINAL_STATUSES = ['草稿', '待生效', '已失败']

// 同一模型的训练执行锁：竞态提交直接合并，不会生成第二份结果。
const trainingLocks = new Set<string>()

export type TrainOutcome = ActionResult & { resultId?: string }

export type ReportForecast = {
  modelCode: string
  usable: boolean
  resultId: string
  batch: string
  datasetCode: string
  interval: string
  sampleCount: string
  note: string
}

function splitFields(text: string): string[] {
  return String(text ?? '')
    .split(/[\uFF0C,、]/)
    .map((item) => item.trim())
    .filter((item) => item !== '')
}

function findByCode(key: string, field: string, code: string): EntryRow | undefined {
  return listRows(key).find((row) => String(row[field] ?? '') === code)
}

function resultRows(): EntryRow[] {
  return listRows(RESULT_KEY)
}

function nextResultId(rows: EntryRow[]): { id: number; code: string } {
  let maxId = 0
  let maxNum = 0
  for (const row of rows) {
    maxId = Math.max(maxId, Number(row.id) || 0)
    const match = String(row['结果编号'] ?? '').match(/^FRUN-(\d+)$/)
    if (match) {
      maxNum = Math.max(maxNum, Number(match[1]))
    }
  }
  return { id: maxId + 1, code: `FRUN-${String(maxNum + 1).padStart(4, '0')}` }
}

function nextBatchSeq(rows: EntryRow[], modelCode: string, datasetCode: string): number {
  return (
    rows.filter((row) => String(row['模型编号']) === modelCode && String(row['数据集版本']) === datasetCode)
      .length + 1
  )
}

function latestEffective(rows: EntryRow[], modelCode: string): EntryRow | undefined {
  return rows
    .filter((row) => String(row['模型编号']) === modelCode && String(row.status) === '已生效')
    .sort((a, b) => String(b['结果编号']).localeCompare(String(a['结果编号'])))[0]
}

// 冲突判定：最新数据集样本不足时，「保留旧模型」与「产出不足草稿」两种做法都成立，
// 这里把两种做法的口径都记下来，冲突时可直接比较；系统默认产出带不足标记的草稿。
function conflictNote(previousCode: string, draftCode: string, basis: string): string {
  const keep = `做法一·保留旧模型：不生成新结果，报告继续读取${previousCode || '无'}，预测停留在旧训练范围`
  const draft = `做法二·不足草稿：生成${draftCode}（${basis}），报告仍读取${previousCode || '无'}，草稿仅留痕不进报告`
  return `${keep}；${draft}。当前采用做法二，两种做法报告口径一致`
}

function upsertResult(rows: EntryRow[], result: EntryRow): EntryRow[] {
  const index = rows.findIndex((row) => Number(row.id) === Number(result.id))
  const next = [...rows]
  if (index >= 0) {
    next[index] = result
  } else {
    next.push(result)
  }
  saveRows(RESULT_KEY, next)
  return next
}

function baseResult(model: EntryRow, datasetCode: string, batch: string, resultCode: string, id: number): EntryRow {
  return {
    id,
    status: '草稿',
    pending: true,
    abnormal: false,
    结果编号: resultCode,
    模型编号: String(model['模型编号']),
    训练批次: batch,
    数据集版本: datasetCode,
    特征字段: String(model['特征字段']),
    样本量: 0,
    预测区间: '',
    样本依据: '',
    异常原因: '',
    预测步长: String(model['预测步长']),
    预测明细: '',
    冲突判定: '',
  }
}

// 训练执行：幂等（同模型 + 同数据集版本的非终态结果原地更新），竞态请求被锁合并。
export function executeTraining(modelCode: string): TrainOutcome {
  if (trainingLocks.has(modelCode)) {
    return { ok: false, message: `模型 ${modelCode} 训练执行中，竞态请求已合并，只保留一份预测模型结果` }
  }
  trainingLocks.add(modelCode)
  try {
    const model = findByCode(MODEL_KEY, '模型编号', modelCode)
    if (!model) {
      return { ok: false, message: `没有找到模型编号为 ${modelCode} 的预测模型` }
    }
    const datasetCode = String(model['训练数据集'] ?? '')
    const rows = resultRows()
    const existing = rows.find(
      (row) =>
        String(row['模型编号']) === modelCode &&
        String(row['数据集版本']) === datasetCode &&
        NON_FINAL_STATUSES.includes(String(row.status)),
    )
    const id = existing ? Number(existing.id) : nextResultId(rows).id
    const resultCode = existing ? String(existing['结果编号']) : nextResultId(rows).code
    const batch = existing
      ? String(existing['训练批次'])
      : makeBatchCode(modelCode, datasetCode, nextBatchSeq(rows, modelCode, datasetCode))
    const result = baseResult(model, datasetCode, batch, resultCode, id)

    const fail = (reason: string): TrainOutcome => {
      result.status = '已失败'
      result.abnormal = true
      result['异常原因'] = reason
      upsertResult(rows, result)
      return { ok: false, resultId: resultCode, message: `训练中断：${reason}。结果 ${resultCode} 已标记已失败，可重试` }
    }

    const dataset = findByCode(DATASET_KEY, '数据集编号', datasetCode)
    if (!dataset) {
      return fail(`训练数据集 ${datasetCode} 未登记`)
    }
    if (String(dataset.status) !== '已生效') {
      return fail(`训练数据集 ${datasetCode} 当前状态「${dataset.status}」，未生效`)
    }
    const features = orderFeaturesOf(datasetCode)
    if (features.length === 0) {
      return fail(`数据集 ${datasetCode} 没有可读的历史订单特征`)
    }

    const step = Number.parseInt(String(model['预测步长']), 10) || 7
    const featureFields = splitFields(String(model['特征字段'] ?? ''))
    const datasetFields = splitFields(String(dataset['字段列表'] ?? ''))
    const assess = assessSamples(features, featureFields, datasetFields, String(model['目标字段'] ?? ''), step)
    result['样本量'] = assess.sampleCount

    if (!assess.sufficient) {
      // 阻断发布：保留上一份可用模型，产出带不足标记的草稿，并记录两种做法的冲突判定。
      const previous = latestEffective(rows, modelCode)
      result.status = '草稿'
      result.abnormal = true
      result['样本依据'] = assess.basis
      result['冲突判定'] = conflictNote(previous ? String(previous['结果编号']) : '', resultCode, assess.basis)
      upsertResult(rows, result)
      return {
        ok: false,
        resultId: resultCode,
        message: `${assess.basis}，已阻断发布并保留上一份可用模型${previous ? `（${previous['结果编号']}）` : ''}；本次产出带不足标记的草稿 ${resultCode}`,
      }
    }

    const points = trainPredict(features, step, String(model['置信区间'] ?? ''), featureFields)
    if (points.length === 0) {
      return fail('有效样本不足以拟合预测曲线')
    }
    result.status = '待生效'
    result['样本依据'] = assess.basis
    result['预测区间'] = predictInterval(points)
    result['预测明细'] = JSON.stringify(points)
    upsertResult(rows, result)
    return {
      ok: true,
      resultId: resultCode,
      message: `训练完成：结果 ${resultCode} 待生效，批次 ${batch}，样本 ${assess.sampleCount} 条，预测区间 ${result['预测区间']}`,
    }
  } finally {
    trainingLocks.delete(modelCode)
  }
}

function findResult(resultId: string): EntryRow | undefined {
  return resultRows().find((row) => String(row['结果编号']) === resultId)
}

// 生效发布：样本不足 / 特征缺失 / 已失败一律阻断；发布成功则上一份已生效结果转为已失效留痕。
export function publishResult(resultId: string): ActionResult {
  const rows = resultRows()
  const row = rows.find((item) => String(item['结果编号']) === resultId)
  if (!row) {
    return { ok: false, message: `没有找到结果编号为 ${resultId} 的预测模型结果` }
  }
  const status = String(row.status)
  if (status === '已生效') {
    return { ok: false, message: `结果 ${resultId} 已经是「已生效」，不用重复发布` }
  }
  if (status === '已失效') {
    return { ok: false, message: `结果 ${resultId} 已失效，不能重新发布，请重新执行训练` }
  }
  if (status === '已失败') {
    return { ok: false, message: `结果 ${resultId} 已失败：${row['异常原因'] || '原因未记录'}。请先重试训练` }
  }
  if (status === '草稿') {
    const previous = latestEffective(rows, String(row['模型编号']))
    return {
      ok: false,
      message: `${row['样本依据'] || '样本依据缺失'}，阻断发布；保留上一份可用模型${previous ? `（${previous['结果编号']}）` : '（当前无）'}`,
    }
  }
  if (!batchMatchesDataset(String(row['训练批次']), String(row['模型编号']), String(row['数据集版本']))) {
    return { ok: false, message: `结果 ${resultId} 训练批次与数据集版本不同源，阻断发布，防止报告引用训练范围外的预测` }
  }
  const modelCode = String(row['模型编号'])
  const superseded: string[] = []
  const next = rows.map((item) => {
    if (
      String(item['模型编号']) === modelCode &&
      String(item.status) === '已生效' &&
      Number(item.id) !== Number(row.id)
    ) {
      superseded.push(String(item['结果编号']))
      return { ...item, status: '已失效', pending: false }
    }
    if (Number(item.id) === Number(row.id)) {
      return { ...item, status: '已生效', pending: false, abnormal: false }
    }
    return item
  })
  saveRows(RESULT_KEY, next)
  const tail = superseded.length > 0 ? `，上一份 ${superseded.join('、')} 已转为已失效留痕` : ''
  return { ok: true, message: `结果 ${resultId} 已生效，报告将读取该预测${tail}` }
}

// 重试训练：已失败 / 不足草稿按原模型重新执行训练，幂等更新同一份结果。
export function retryTraining(resultId: string): TrainOutcome {
  const row = findResult(resultId)
  if (!row) {
    return { ok: false, message: `没有找到结果编号为 ${resultId} 的预测模型结果` }
  }
  if (!['草稿', '已失败'].includes(String(row.status))) {
    return { ok: false, message: `结果 ${resultId} 当前状态「${row.status}」，无需重试` }
  }
  return executeTraining(String(row['模型编号']))
}

export function invalidateResult(resultId: string): ActionResult {
  const rows = resultRows()
  const row = rows.find((item) => String(item['结果编号']) === resultId)
  if (!row) {
    return { ok: false, message: `没有找到结果编号为 ${resultId} 的预测模型结果` }
  }
  if (String(row.status) === '已失效') {
    return { ok: false, message: `结果 ${resultId} 已经是「已失效」，不用重复操作` }
  }
  const wasEffective = String(row.status) === '已生效'
  const next = rows.map((item) =>
    Number(item.id) === Number(row.id) ? { ...item, status: '已失效', pending: false } : item,
  )
  saveRows(RESULT_KEY, next)
  return {
    ok: true,
    message: `结果 ${resultId} 已标记失效${wasEffective ? '，报告将不再引用该预测' : ''}`,
  }
}

// 报告读取口径：只取「已生效」且训练批次与数据集版本同源的结果；
// 不合格草稿、待生效、已失败、批次不同源的结果都不会被报告引用。
export function reportForecast(modelCode: string): ReportForecast {
  const effective = resultRows().filter(
    (row) => String(row['模型编号']) === modelCode && String(row.status) === '已生效',
  )
  const usable = effective.filter((row) =>
    batchMatchesDataset(String(row['训练批次']), modelCode, String(row['数据集版本'])),
  )
  const dropped = effective.length - usable.length
  const latest = usable.sort((a, b) => String(b['结果编号']).localeCompare(String(a['结果编号'])))[0]
  if (!latest) {
    return {
      modelCode,
      usable: false,
      resultId: '',
      batch: '',
      datasetCode: '',
      interval: '',
      sampleCount: '',
      note: '无已生效且批次同源的预测结果，报告暂不引用',
    }
  }
  return {
    modelCode,
    usable: true,
    resultId: String(latest['结果编号']),
    batch: String(latest['训练批次']),
    datasetCode: String(latest['数据集版本']),
    interval: String(latest['预测区间']),
    sampleCount: String(latest['样本量']),
    note: dropped > 0 ? `${dropped} 份已生效结果批次不同源，报告已弃用` : '批次与数据集版本同源，可引用',
  }
}

export function listReportForecasts(): ReportForecast[] {
  return listRows(MODEL_KEY).map((model) => reportForecast(String(model['模型编号'])))
}

export function resultDetail(resultId: string): EntryRow | undefined {
  return findResult(resultId)
}

export function resultPoints(row: EntryRow | undefined): PredictionPoint[] {
  if (!row) {
    return []
  }
  const raw = String(row['预测明细'] ?? '')
  if (raw === '') {
    return []
  }
  try {
    return JSON.parse(raw) as PredictionPoint[]
  } catch {
    return []
  }
}

function csvCell(value: string): string {
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value
}

// 单份预测模型结果的可下载文件：元信息 + 预测明细，与列表条目同源。
export function downloadResultFile(resultId: string): ActionResult {
  const row = findResult(resultId)
  if (!row) {
    return { ok: false, message: `没有找到结果编号为 ${resultId} 的预测模型结果` }
  }
  const metaLines = [
    ['结果编号', String(row['结果编号'])],
    ['模型编号', String(row['模型编号'])],
    ['训练批次', String(row['训练批次'])],
    ['数据集版本', String(row['数据集版本'])],
    ['特征字段', String(row['特征字段'])],
    ['样本量', String(row['样本量'])],
    ['样本依据', String(row['样本依据'])],
    ['预测区间', String(row['预测区间'])],
    ['当前状态', String(row.status)],
    ['异常原因', String(row['异常原因'] ?? '')],
    ['冲突判定', String(row['冲突判定'] ?? '')],
  ].map(([key, value]) => `${key},${csvCell(value)}`)
  const detailLines = ['日期,预测值,下限,上限']
  for (const point of resultPoints(row)) {
    detailLines.push(`${point.日期},${point.预测值},${point.下限},${point.上限}`)
  }
  const content = `\uFEFF预测模型结果\n${metaLines.join('\n')}\n\n预测明细\n${detailLines.join('\n')}`
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `预测结果-${resultId}.csv`
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
  return { ok: true, message: `结果 ${resultId} 的预测文件已生成下载` }
}
