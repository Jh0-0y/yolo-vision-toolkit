import { useMemo, useState } from 'react'
import { Badge, Button, Card, Group, Loader, SimpleGrid, Stack, Text } from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { predict, type ImageItem, type PredictResponse } from '../../api/client'
import type { TestConfig } from './TestControls'
import BoxOverlay from './BoxOverlay'
import DatasetPicker from './DatasetPicker'

type Cell = { img: ImageItem; result?: PredictResponse; error?: string }

export default function BatchMode({ projectId, cfg }: { projectId: string; cfg: TestConfig }) {
  const [cells, setCells] = useState<Cell[]>([])
  const [running, setRunning] = useState(false)
  const [pickerOpen, setPickerOpen] = useState(false)

  async function runAll(imgs: ImageItem[]) {
    if (!cfg.selected.length) {
      notifications.show({ color: 'red', message: '모델을 하나 이상 선택하세요' })
      return
    }
    setCells(imgs.map((img) => ({ img })))
    setRunning(true)
    // backend serializes on a single worker; fire concurrently and let it queue
    const done = await Promise.all(
      imgs.map(async (img): Promise<Cell> => {
        try {
          const result = await predict({
            modelIds: cfg.selected,
            projectId,
            params: { conf: cfg.conf, iou_wbf: cfg.iou, imgsz: Number(cfg.imgsz), device: cfg.device === 'auto' ? null : cfg.device },
            imageProjectId: projectId,
            imageName: img.name,
          })
          return { img, result }
        } catch (e) {
          return { img, error: (e as Error).message }
        }
      }),
    )
    setCells(done)
    setRunning(false)
  }

  const totalBoxes = useMemo(
    () => cells.reduce((n, c) => n + (c.result?.boxes.filter((b) => b.score >= cfg.conf).length ?? 0), 0),
    [cells, cfg.conf],
  )

  return (
    <Card withBorder radius="md" padding="md">
      <Group justify="space-between" mb="md">
        <Group gap="xs">
          <Button onClick={() => setPickerOpen(true)}>데이터셋에서 여러 장 선택</Button>
          {cells.length > 0 && (
            <Text c="dimmed" size="sm">
              {cells.length}장 · 검출 {totalBoxes}개 (conf ≥ {cfg.conf.toFixed(2)})
            </Text>
          )}
        </Group>
        {running && <Loader size="sm" />}
      </Group>

      {cells.length === 0 ? (
        <Text c="dimmed" size="sm">
          여러 이미지를 골라 한 번에 추론하고 결과를 그리드로 훑어보세요.
        </Text>
      ) : (
        <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="sm">
          {cells.map((c) => {
            const boxes = (c.result?.boxes ?? []).filter((b) => b.score >= cfg.conf)
            return (
              <Stack key={c.img.name} gap={4}>
                <BoxOverlay src={c.img.url} boxes={boxes} showLabels={cfg.showLabels} />
                <Group justify="space-between" gap="xs">
                  <Text size="xs" truncate="end" maw={180}>
                    {c.img.name}
                  </Text>
                  {c.error ? (
                    <Badge size="xs" color="red" variant="light">
                      오류
                    </Badge>
                  ) : c.result ? (
                    <Badge size="xs" color="teal" variant="light">
                      {boxes.length} boxes
                    </Badge>
                  ) : (
                    <Loader size="xs" />
                  )}
                </Group>
              </Stack>
            )
          })}
        </SimpleGrid>
      )}

      <DatasetPicker
        projectId={projectId}
        opened={pickerOpen}
        onClose={() => setPickerOpen(false)}
        onPick={() => {}}
        multiple
        onPickMany={(imgs) => {
          setPickerOpen(false)
          void runAll(imgs)
        }}
      />
    </Card>
  )
}
