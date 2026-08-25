// 텐서보드가 왼쪽에서 런을 켜고 끄듯, 계열을 켜고 끈다.
//
// **무엇을 켜는지는 모른다.** 목록·색·체크 상태와 아래 컨트롤 슬롯만 다루므로,
// 벤치마크의 엔트리든 나중에 학습의 지표 계열이든 그대로 쓸 수 있다.
import {
  Card,
  Checkbox,
  Group,
  Stack,
  Text,
} from '@mantine/core'


export interface Series {
  id: string
  label: string
  color: string
  hint?: string
}

export default function SeriesRail({
  title,
  series,
  enabled,
  onToggle,
  children,
}: {
  title: string
  series: Series[]
  enabled: Set<string>
  onToggle: (id: string) => void
  children?: React.ReactNode
}) {
  return (
    <Card withBorder radius="md" padding="sm" style={{ minWidth: 220, alignSelf: 'flex-start' }}>
      <Text size="xs" fw={700} c="dimmed" tt="uppercase" mb="xs">
        {title}
      </Text>
      <Stack gap={6}>
        {series.map((s) => (
          <Group key={s.id} gap={8} wrap="nowrap" align="flex-start">
            <Checkbox
              size="xs"
              checked={enabled.has(s.id)}
              onChange={() => onToggle(s.id)}
              styles={{ input: { cursor: 'pointer' } }}
            />
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 2,
                background: s.color,
                flexShrink: 0,
                marginTop: 4,
              }}
            />
            <div style={{ minWidth: 0 }}>
              <Text size="xs" lineClamp={2}>{s.label}</Text>
              {s.hint && <Text size="xs" c="dimmed">{s.hint}</Text>}
            </div>
          </Group>
        ))}
      </Stack>
      {children && <div style={{ marginTop: 16 }}>{children}</div>}
    </Card>
  )
}
