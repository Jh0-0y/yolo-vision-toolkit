import {
  useMemo,
  useState,
} from 'react'
import {
  Card,
  Group,
  MultiSelect,
  SegmentedControl,
  Text,
} from '@mantine/core'
import {
  LineChart,
} from '@mantine/charts'
import {
  type PerClassEpoch,
} from '../../api/client'
import {
  classColor,
} from '../../stores/editorStore'


type PcMetric = 'mAP50-95' | 'mAP50' | 'precision' | 'recall'
const PC_METRICS: { value: PcMetric; label: string }[] = [
  { value: 'mAP50-95', label: 'mAP50-95' },
  { value: 'mAP50', label: 'mAP50' },
  { value: 'precision', label: 'P' },
  { value: 'recall', label: 'R' },
]

export default function PerClassEpochChart({ history }: { history: PerClassEpoch[] }) {
  const [metric, setMetric] = useState<PcMetric>('mAP50-95')
  const [picked, setPicked] = useState<string[]>([])

  // class identity/order from the last epoch's snapshot
  const classes = history[history.length - 1]?.metrics ?? []

  const shown = useMemo(() => {
    if (picked.length) return classes.filter((c) => picked.includes(String(c.cls)))
    if (classes.length > 8) {
      return [...classes].sort((a, b) => (b[metric] as number) - (a[metric] as number)).slice(0, 8)
    }
    return classes
  }, [classes, picked, metric])

  const data = useMemo(
    () =>
      history.map((h) => {
        const row: Record<string, number> = { epoch: h.epoch }
        for (const c of shown) {
          const m = h.metrics.find((x) => x.cls === c.cls)
          if (m) row[c.name] = m[metric] as number
        }
        return row
      }),
    [history, shown, metric],
  )

  const series = shown.map((c) => ({ name: c.name, color: classColor(c.cls) }))

  return (
    <Card withBorder radius="md" padding="md">
      <Group justify="space-between" wrap="wrap" mb="xs" gap="xs">
        <Text size="sm" fw={600}>
          Per-class over epochs
        </Text>
        <Group gap="xs">
          <SegmentedControl
            size="xs"
            value={metric}
            onChange={(v) => setMetric(v as PcMetric)}
            data={PC_METRICS}
          />
          <MultiSelect
            size="xs"
            w={220}
            placeholder={
              picked.length ? undefined : classes.length > 8 ? 'Top 8 (pick classes)' : 'All classes'
            }
            data={classes.map((c) => ({ value: String(c.cls), label: c.name }))}
            value={picked}
            onChange={setPicked}
            clearable
            searchable
            maxDropdownHeight={220}
          />
        </Group>
      </Group>
      <LineChart
        h={240}
        data={data}
        dataKey="epoch"
        series={series}
        curveType="monotone"
        withDots={data.length < 40}
        withLegend
        yAxisProps={{ domain: [0, 1] }}
        valueFormatter={(v) => v.toFixed(3)}
      />
    </Card>
  )
}

