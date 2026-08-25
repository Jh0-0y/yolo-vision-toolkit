// 혼동행렬 — 행이 실제, 열이 예측. 마지막 행·열은 background 다.
//
// 대각선은 맞힌 것, 마지막 **열**은 놓침, 마지막 **행**은 헛것이다.
// background×background 는 뜻이 없어 비어 있다(null).
import {
  Table,
  Text,
  useMantineColorScheme,
} from '@mantine/core'


export default function ConfusionMatrix({
  labels,
  rows,
}: {
  labels: string[]
  rows: (number | null)[][]
}) {
  const { colorScheme } = useMantineColorScheme()
  const dark = colorScheme === 'dark'
  const max = Math.max(1, ...rows.flat().map((v) => v ?? 0))

  // 값이 클수록 진하게. 라이트·다크 양쪽에서 글자가 읽히도록 명도를 반대로 준다.
  const cell = (v: number | null) => {
    if (v === null) return { background: 'transparent' }
    const a = v / max
    return {
      background: dark
        ? `rgba(77, 171, 247, ${0.08 + a * 0.55})`
        : `rgba(34, 139, 230, ${0.06 + a * 0.45})`,
    }
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <Table withTableBorder withColumnBorders verticalSpacing={4} horizontalSpacing={8}>
        <Table.Thead>
          <Table.Tr>
            <Table.Th />
            {labels.map((l) => (
              <Table.Th key={l} style={{ whiteSpace: 'nowrap' }}>
                <Text size="xs" c="dimmed">{l}</Text>
              </Table.Th>
            ))}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map((r, i) => (
            <Table.Tr key={labels[i]}>
              <Table.Th style={{ whiteSpace: 'nowrap' }}>
                <Text size="xs" c="dimmed">{labels[i]}</Text>
              </Table.Th>
              {r.map((v, j) => (
                <Table.Td key={j} style={{ textAlign: 'right', ...cell(v) }}>
                  <Text size="xs">{v === null ? '—' : v.toLocaleString()}</Text>
                </Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      <Text size="xs" c="dimmed" mt={6}>
        Rows: actual · Columns: predicted. Last column = missed, last row = false alarms.
      </Text>
    </div>
  )
}
