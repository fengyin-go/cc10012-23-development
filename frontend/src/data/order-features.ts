// 历史订单特征数据层：训练执行从这里读取各数据集版本的特征行。
// 纯前端没有真实数仓，这里用确定性生成器产出可核对的历史订单特征：
//   DATA-0001 历史订单主集：60 天完整特征
//   DATA-0002 历史订单上一版：45 天完整特征（上一份完整预测的来源）
//   DATA-0003 促销特征增量集：只带来促销特征更新，历史窗只有 5 天（验收场景）

export type OrderFeatureRow = {
  日期: string
  订单量: number | null
  促销标记: number | null
  客单价: number | null
  流量: number | null
}

function hashSeed(text: string): number {
  let hash = 2166136261
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return hash >>> 0
}

// mulberry32：确定性伪随机，同一个数据集版本每次生成的特征行完全一致。
function mulberry32(seed: number): () => number {
  let state = seed
  return () => {
    state = (state + 0x6d2b79f5) | 0
    let t = Math.imul(state ^ (state >>> 15), 1 | state)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function shiftDay(day: string, delta: number): string {
  const date = new Date(`${day}T00:00:00Z`)
  date.setUTCDate(date.getUTCDate() + delta)
  return date.toISOString().slice(0, 10)
}

function buildRows(datasetCode: string, days: number, endDay: string, promoOnly: boolean): OrderFeatureRow[] {
  const rand = mulberry32(hashSeed(datasetCode))
  const rows: OrderFeatureRow[] = []
  const startDay = shiftDay(endDay, -(days - 1))
  for (let i = 0; i < days; i += 1) {
    const day = shiftDay(startDay, i)
    const dow = new Date(`${day}T00:00:00Z`).getUTCDay()
    const promo = dow === 5 || dow === 6 ? 1 : 0
    const seasonal = 18 * Math.sin((dow / 7) * Math.PI * 2)
    const noise = (rand() - 0.5) * 16
    const volume = Math.round((120 + seasonal + promo * 30 + noise) * 10) / 10
    rows.push({
      日期: day,
      订单量: volume,
      促销标记: promo,
      客单价: promoOnly ? null : Math.round((68 + (rand() - 0.5) * 10) * 100) / 100,
      流量: promoOnly ? null : Math.round(2400 + (rand() - 0.5) * 600),
    })
  }
  return rows
}

const ORDER_FEATURES: Record<string, OrderFeatureRow[]> = {
  'DATA-0001': buildRows('DATA-0001', 60, '2026-11-16', false),
  'DATA-0002': buildRows('DATA-0002', 45, '2026-11-10', false),
  // DATA-0003 只带来促销特征更新：客单价、流量没有随行更新，历史窗也只有 5 天。
  'DATA-0003': buildRows('DATA-0003', 5, '2026-11-16', true),
}

export function orderFeaturesOf(datasetCode: string): OrderFeatureRow[] {
  return ORDER_FEATURES[datasetCode] ?? []
}
