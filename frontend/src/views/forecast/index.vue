<template>
  <section class="page" data-module="forecast">
    <header class="page-head">
      <div>
        <h2>趋势预测管理</h2>
        <p class="page-desc">维护趋势预测模型的输入、执行约束与结果记录，为趋势分析提供可验证的数据工程流程。</p>
      </div>
      <div class="page-actions">
        <button class="btn primary" type="button" @click="openCreate">登记预测模型</button>
        <button class="btn" type="button" @click="goResults">查看预测模型结果</button>
        <button class="btn" type="button" @click="exportRows">导出趋势预测清单</button>
      </div>
    </header>

    <div class="stat-row">
      <article v-for="item in stats" :key="item.label" class="stat-card">
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
          <td v-for="column in columns" :key="column">{{ row[column] ?? '—' }}</td>
          <td>{{ row.status }}</td>
          <td class="row-actions">
            <button
              class="link"
              type="button"
              :disabled="training"
              @click="runTraining(row)"
            >
              执行训练
            </button>
            <button
              v-for="action in actions"
              :key="action"
              class="link"
              type="button"
              @click="runAction(action, row)"
            >
              {{ action }}
            </button>
          </td>
        </tr>
        <tr v-if="!rows.length">
          <td :colspan="columns.length + 2" class="empty-state">暂无趋势预测数据，可先登记预测模型</td>
        </tr>
      </tbody>
    </table>

    <section class="report-panel">
      <h3>报告读取口径（报告只引用「已生效」且训练批次与数据集版本同源的预测）</h3>
      <table class="data-table">
        <thead>
          <tr>
            <th>模型编号</th>
            <th>生效结果</th>
            <th>训练批次</th>
            <th>数据集版本</th>
            <th>样本量</th>
            <th>预测区间</th>
            <th>说明</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="item in reportRows" :key="item.modelCode">
            <td>{{ item.modelCode }}</td>
            <td>{{ item.usable ? item.resultId : '—' }}</td>
            <td>{{ item.usable ? item.batch : '—' }}</td>
            <td>{{ item.usable ? item.datasetCode : '—' }}</td>
            <td>{{ item.usable ? item.sampleCount : '—' }}</td>
            <td>{{ item.usable ? item.interval : '—' }}</td>
            <td>{{ item.note }}</td>
          </tr>
          <tr v-if="!reportRows.length">
            <td colspan="7" class="empty-state">暂无预测模型</td>
          </tr>
        </tbody>
      </table>
    </section>

    <footer class="page-foot">
      <span>共 {{ total }} 条趋势预测记录</span>
      <span v-if="errorMessage" class="error-text">{{ errorMessage }}</span>
      <span v-else-if="infoMessage" class="ok-text">{{ infoMessage }}</span>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import {
  downloadEntries,
  listEntries,
  moduleMeta,
  runAction as applyAction,
} from '@/api/local-service'
import { executeTraining, listReportForecasts, type ReportForecast } from '@/api/forecast-training'
import type { EntryRow } from '@/data/types'

const meta = moduleMeta('forecast')
const router = useRouter()
const columns = ["模型编号", "模型类型", "训练数据集", "目标字段", "特征字段", "预测步长", "置信区间", "评估指标"]
const actions = ["提交训练", "生效发布", "标记失效"]
const statuses = ["草稿", "待训练", "已生效", "已失效"]
const stats = [{"label": "预测模型总数", "value": 0}, {"label": "已生效模型", "value": 0}, {"label": "待训练模型", "value": 0}]

const rows = ref<EntryRow[]>([])
const total = ref(0)
const errorMessage = ref('')
const infoMessage = ref('')
const filters = ref<Record<string, string>>({})
const filterFields = columns.slice(0, 3)
const training = ref(false)
const reportRows = ref<ReportForecast[]>([])
const statusSummary = computed(() =>
  statuses.map((status: string) => ({
    status,
    count: rows.value.filter((row) => String(row.status) === status).length,
  })),
)

function resetFilters() {
  filters.value = {}
  reload()
}

function exportRows() {
  downloadEntries(meta.key)
}

function goResults() {
  router.push('/forecastrun')
}

function openCreate() {
  errorMessage.value = '预测模型登记入口尚未接入审批流'
}

function runTraining(row: EntryRow) {
  training.value = true
  try {
    const result = executeTraining(String(row['模型编号']))
    reload()
    if (result.ok) {
      infoMessage.value = `${result.message}；请到「预测模型结果」生效发布`
    } else {
      errorMessage.value = result.message
    }
  } finally {
    training.value = false
  }
}

function runAction(action: string, row: EntryRow) {
  errorMessage.value = ''
  const result = applyAction(meta.key, Number(row.id), action)
  if (!result.ok) {
    errorMessage.value = result.message
    return
  }
  reload()
}

function reload() {
  errorMessage.value = ''
  infoMessage.value = ''
  try {
    const payload = listEntries(meta.key, filters.value)
    rows.value = payload.items
    total.value = payload.total
    reportRows.value = listReportForecasts()
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '趋势预测列表读取失败'
  }
}

onMounted(reload)
</script>
