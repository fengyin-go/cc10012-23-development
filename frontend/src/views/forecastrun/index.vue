<template>
  <section class="page" data-module="forecastrun">
    <header class="page-head">
      <div>
        <h2>预测模型结果管理</h2>
        <p class="page-desc">承载训练执行产出的预测模型结果业务数据：训练批次、数据集版本、样本依据与预测明细同源可核对，报告只读取已生效且批次同源的预测。</p>
      </div>
      <div class="page-actions">
        <button class="btn" type="button" @click="exportRows">导出预测模型结果清单</button>
      </div>
    </header>

    <div class="stat-row">
      <article v-for="item in statCards" :key="item.label" class="stat-card">
        <span class="stat-label">{{ item.label }}</span>
        <strong class="stat-value">{{ item.value }}</strong>
      </article>
    </div>

    <p class="status-legend">
      <span v-for="item in statusSummary" :key="item.status" class="legend-item">
        {{ item.status }}：{{ item.count }}
      </span>
    </p>

    <form class="filter-bar" @submit.prevent="reload">
      <label v-for="field in filterFields" :key="field" class="filter-item">
        <span>{{ field }}</span>
        <input v-model="filters[field]" :placeholder="`按${field}检索`" />
      </label>
      <button class="btn" type="submit">查询</button>
      <button class="btn ghost" type="button" @click="resetFilters">重置条件</button>
    </form>

    <table class="data-table">
      <thead>
        <tr>
          <th v-for="column in columns" :key="column">{{ column }}</th>
          <th>当前状态</th>
          <th>可执行动作</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="String(row.id)">
          <td v-for="column in columns" :key="column">{{ row[column] === '' || row[column] === undefined ? '—' : row[column] }}</td>
          <td>
            <span class="badge" :class="badgeClass(String(row.status))">{{ row.status }}</span>
          </td>
          <td class="row-actions">
            <button
              v-if="['草稿', '待生效', '已失败'].includes(String(row.status))"
              class="link"
              type="button"
              @click="runPublish(row)"
            >
              生效发布
            </button>
            <button
              v-if="['草稿', '已失败'].includes(String(row.status))"
              class="link"
              type="button"
              @click="runRetry(row)"
            >
              重试训练
            </button>
            <button
              v-if="String(row.status) !== '已失效'"
              class="link"
              type="button"
              @click="runInvalidate(row)"
            >
              标记失效
            </button>
            <button class="link" type="button" @click="runDownload(row)">下载文件</button>
            <button class="link" type="button" @click="toggleDetail(row)">
              {{ detailId === String(row['结果编号']) ? '收起' : '详情' }}
            </button>
          </td>
        </tr>
        <tr v-if="!rows.length">
          <td :colspan="columns.length + 2" class="empty-state">暂无预测模型结果，请先在「趋势预测」执行训练</td>
        </tr>
      </tbody>
    </table>

    <section v-if="detailRow" class="detail-panel">
      <h3>结果详情：{{ detailRow['结果编号'] }}（{{ detailRow.status }}）</h3>
      <p class="muted">
        模型 {{ detailRow['模型编号'] }} · 批次 {{ detailRow['训练批次'] }} · 数据集版本 {{ detailRow['数据集版本'] }} ·
        样本量 {{ detailRow['样本量'] }} · 预测区间 {{ detailRow['预测区间'] || '—' }}
      </p>
      <p v-if="detailRow['样本依据']" class="muted">样本依据：{{ detailRow['样本依据'] }}</p>
      <p v-if="detailRow['异常原因']" class="error-text">异常原因：{{ detailRow['异常原因'] }}</p>
      <p v-if="detailRow['冲突判定']" class="muted">冲突判定：{{ detailRow['冲突判定'] }}</p>
      <table v-if="detailPoints.length" class="data-table">
        <thead>
          <tr><th>日期</th><th>预测值</th><th>下限</th><th>上限</th></tr>
        </thead>
        <tbody>
          <tr v-for="point in detailPoints" :key="point.日期">
            <td>{{ point.日期 }}</td>
            <td>{{ point.预测值 }}</td>
            <td>{{ point.下限 }}</td>
            <td>{{ point.上限 }}</td>
          </tr>
        </tbody>
      </table>
      <p v-else class="muted">该结果暂无预测明细（样本不足草稿或已失败结果不生成预测值）</p>
    </section>

    <footer class="page-foot">
      <span>共 {{ total }} 条预测模型结果记录</span>
      <span v-if="errorMessage" class="error-text">{{ errorMessage }}</span>
      <span v-else-if="infoMessage" class="ok-text">{{ infoMessage }}</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

import {
  downloadEntries,
  listEntries,
  moduleMeta,
} from '@/api/local-service'
import {
  downloadResultFile,
  invalidateResult,
  publishResult,
  resultDetail,
  resultPoints,
  retryTraining,
} from '@/api/forecast-training'
import type { EntryRow } from '@/data/types'
import type { PredictionPoint } from '@/data/forecast-engine'

const meta = moduleMeta('forecastrun')
const columns = ["结果编号", "模型编号", "训练批次", "数据集版本", "特征字段", "样本量", "预测区间", "样本依据", "异常原因"]
const statuses = ["草稿", "待生效", "已生效", "已失效", "已失败"]

const rows = ref<EntryRow[]>([])
const total = ref(0)
const errorMessage = ref('')
const infoMessage = ref('')
const filters = ref<Record<string, string>>({})
const filterFields = columns.slice(0, 3)
const detailId = ref('')
const detailRow = ref<EntryRow | undefined>(undefined)
const detailPoints = ref<PredictionPoint[]>([])

const statCards = computed(() => [
  { label: '预测结果总数', value: rows.value.length },
  { label: '已生效结果', value: rows.value.filter((row) => String(row.status) === '已生效').length },
  { label: '待生效结果', value: rows.value.filter((row) => String(row.status) === '待生效').length },
])

const statusSummary = computed(() =>
  statuses.map((status: string) => ({
    status,
    count: rows.value.filter((row) => String(row.status) === status).length,
  })),
)

function badgeClass(status: string): string {
  if (status === '已生效') return 'ok'
  if (status === '已失败') return 'err'
  if (status === '草稿') return 'warn'
  return ''
}

function resetFilters() {
  filters.value = {}
  reload()
}

function exportRows() {
  downloadEntries(meta.key)
}

function showResult(result: { ok: boolean; message: string }) {
  reload()
  if (result.ok) {
    infoMessage.value = result.message
  } else {
    errorMessage.value = result.message
  }
}

function runPublish(row: EntryRow) {
  showResult(publishResult(String(row['结果编号'])))
}

function runRetry(row: EntryRow) {
  showResult(retryTraining(String(row['结果编号'])))
}

function runInvalidate(row: EntryRow) {
  showResult(invalidateResult(String(row['结果编号'])))
}

function runDownload(row: EntryRow) {
  showResult(downloadResultFile(String(row['结果编号'])))
}

function toggleDetail(row: EntryRow) {
  const code = String(row['结果编号'])
  if (detailId.value === code) {
    detailId.value = ''
    detailRow.value = undefined
    detailPoints.value = []
    return
  }
  detailId.value = code
  detailRow.value = resultDetail(code)
  detailPoints.value = resultPoints(detailRow.value)
}

function reload() {
  errorMessage.value = ''
  infoMessage.value = ''
  try {
    const payload = listEntries(meta.key, filters.value)
    rows.value = payload.items
    total.value = payload.total
    if (detailId.value) {
      detailRow.value = resultDetail(detailId.value)
      detailPoints.value = resultPoints(detailRow.value)
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '预测模型结果列表读取失败'
  }
}

onMounted(reload)
</script>
