// 검수완료 이미지를 타일로 쪼개 **새 데이터셋**을 만든다.
//
// 미리보기가 두 단계다. 격자(`4×2 = 8 tiles/image`)는 브라우저 산술이라 노브를
// 만질 때마다 즉시 바뀌고, 포지티브/네거티브 장수는 서버가 이미지 크기와 라벨을
// 훑어야 알 수 있어 [Estimate] 를 눌렀을 때만 돈다. 네거티브 비율은 그 결과에
// 곱하기만 하면 되므로 슬라이더를 움직여도 서버를 다시 때리지 않는다.
import { useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Group,
  NumberInput,
  Slider,
  Stack,
  Text,
  TextInput,
  ThemeIcon,
} from '@mantine/core'
import { notifications } from '@mantine/notifications'
import { IconGridDots } from '@tabler/icons-react'
import { useMutation } from '@tanstack/react-query'
import {
  createTiledDataset,
  estimateTiling,
  type DatasetOut,
  type TileEstimate,
} from '../../api/client'
import { useJobStore } from '../../stores/jobStore'

interface Props {
  projectId: string
  datasetId: string
  dataset: DatasetOut
  onStarted?: () => void
}

// 기본값은 백엔드와 같아야 한다 — 화면이 보여주는 격자가 서버가 만드는 격자다
const DEFAULTS = { tileSize: 640, stride: 480, minVisibility: 0.6 }

export default function TilingCard({ projectId, datasetId, dataset, onStarted }: Props) {
  const [tileSize, setTileSize] = useState(DEFAULTS.tileSize)
  const [stride, setStride] = useState(DEFAULTS.stride)
  const [minVisibility, setMinVisibility] = useState(DEFAULTS.minVisibility)
  const [ratio, setRatio] = useState(10) // %
  const [keepAll, setKeepAll] = useState(false)
  const [name, setName] = useState('')
  const [est, setEst] = useState<TileEstimate | null>(null)
  const trackTiling = useJobStore((s) => s.trackTiling)

  const overlap = tileSize - stride
  // 격자를 흔드는 노브가 바뀌면 추정치는 더 이상 이 설정의 것이 아니다
  useEffect(() => setEst(null), [tileSize, stride, minVisibility])

  const params = {
    tile_size: tileSize,
    stride,
    min_visibility: minVisibility,
    negative_ratio: ratio / 100,
    keep_all_negatives: keepAll,
  }

  const estimateRun = useMutation({
    mutationFn: () => estimateTiling(projectId, datasetId, params),
    onSuccess: setEst,
    onError: (e) => {
      setEst(null)
      notifications.show({ message: String(e), color: 'red' })
    },
  })

  // 열자마자 한 번 훑는다 — 격자(`1920×1080 → 4×2 = 8 tiles/image`)를 보려면
  // 이미지 크기를 알아야 하고, 그건 서버만 안다. 이후엔 [Estimate] 로 다시 돈다.
  useEffect(() => {
    if (dataset.reviewed > 0) estimateRun.mutate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const create = useMutation({
    mutationFn: () => createTiledDataset(projectId, datasetId, { ...params, name: name.trim() }),
    onSuccess: (r) => {
      trackTiling(projectId, datasetId, r.dataset_id, `Tiling → ${r.name}`)
      notifications.show({ message: `Building ${r.name}`, color: 'blue' })
      onStarted?.()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  // 네거티브는 서버를 다시 때리지 않고 여기서 곱한다
  const keptNegatives = est
    ? keepAll
      ? est.negative
      : Math.min(est.negative, Math.round(est.positive * (ratio / 100)))
    : 0
  const totalOut = est ? est.positive + keptNegatives : 0
  const dominant = est?.sizes[0]
  // 라벨이 하나도 없는 데이터셋(하드 네거티브)은 막지 않는다 — 전부 담기로 안내만 한다
  const noPositives = est !== null && est.positive === 0

  return (
    <Card withBorder radius="md" padding="lg">
      <Stack gap="md">
        <Group gap="xs">
          <ThemeIcon variant="light" size="lg" radius="md">
            <IconGridDots size={20} />
          </ThemeIcon>
          <div>
            <Text fw={600}>Tile into training patches</Text>
            <Text size="xs" c="dimmed">
              {dataset.reviewed} reviewed images → a new dataset of {tileSize}px tiles
            </Text>
          </div>
        </Group>

        <Group grow align="flex-start">
          <NumberInput
            label="Tile size (px)"
            value={tileSize}
            onChange={(v) => setTileSize(Number(v) || DEFAULTS.tileSize)}
            min={64}
            step={32}
          />
          <NumberInput
            label="Stride (px)"
            value={stride}
            onChange={(v) => setStride(Number(v) || DEFAULTS.stride)}
            min={32}
            max={tileSize}
            step={32}
            error={
              overlap <= 0
                ? 'Stride must be < tile size — boundary objects split in two'
                : undefined
            }
          />
        </Group>

        <Text size="xs" c="dimmed">
          overlap {overlap}px ({tileSize ? Math.round((overlap / tileSize) * 100) : 0}%)
          {dominant
            ? ` · ${dominant.w}×${dominant.h} → ${dominant.cols}×${dominant.rows} = ${
                dominant.cols * dominant.rows
              } tiles/image`
            : ''}
        </Text>

        <NumberInput
          label="Min visibility (0–1)"
          description="Boxes less visible than this are dropped — half-boxes don't poison training"
          value={minVisibility}
          onChange={(v) => setMinVisibility(v === '' || v == null ? DEFAULTS.minVisibility : Number(v))}
          min={0}
          max={1}
          step={0.05}
          decimalScale={2}
        />

        <div>
          <Group justify="space-between">
            <Text size="sm">Negatives (% of positives)</Text>
            <Text size="sm" c="dimmed">
              {keepAll ? 'all' : `${ratio}%`}
            </Text>
          </Group>
          <Slider
            value={ratio}
            onChange={setRatio}
            min={0}
            max={300}
            step={5}
            // 포지티브가 0이면 "포지티브 대비 %" 가 의미를 잃는다 (0의 10% 는 0).
            // 하드 네거티브만 모은 데이터셋이 그렇다 — 전부 담기로 안내한다.
            disabled={keepAll || noPositives}
            marks={[
              { value: 0, label: '0' },
              { value: 100, label: '1:1' },
              { value: 300, label: '3:1' },
            ]}
            mb="md"
          />
          <Checkbox
            label="Keep every negative tile"
            checked={keepAll}
            onChange={(e) => setKeepAll(e.currentTarget.checked)}
            size="xs"
          />
        </div>

        <Group>
          <Button
            variant="light"
            loading={estimateRun.isPending}
            onClick={() => estimateRun.mutate()}
            disabled={overlap <= 0 || dataset.reviewed === 0}
          >
            Estimate
          </Button>
          {est && (
            <Text size="xs" c="dimmed">
              {est.positive.toLocaleString()} positive · {est.negative.toLocaleString()} negative
              → keep {keptNegatives.toLocaleString()} = <b>{totalOut.toLocaleString()} images</b>
            </Text>
          )}
        </Group>

        {noPositives && !keepAll && (
          <Alert color="blue" variant="light" p="xs">
            <Text size="xs">
              No labelled objects in this dataset — every tile is a negative. Turn on
              &quot;keep every negative tile&quot; to bring them all in as hard negatives.
            </Text>
          </Alert>
        )}

        {est && est.undersized > 0 && (
          <Alert color="yellow" variant="light" p="xs">
            <Text size="xs">
              {est.undersized} images are smaller than the tile size — they come through as a
              single tile at their original size (no padding).
            </Text>
          </Alert>
        )}

        <TextInput
          label="New dataset name"
          placeholder={`${dataset.name} (tiled ${tileSize})`}
          value={name}
          onChange={(e) => setName(e.currentTarget.value)}
        />

        <Button
          loading={create.isPending}
          // 서버가 막지 않으므로 여기서 "0장짜리 데이터셋" 만 걸러낸다.
          // 마운트 때 돈 첫 estimate 가 아직 안 끝났으면 급한 클릭을 막는다.
          disabled={
            overlap <= 0 ||
            dataset.reviewed === 0 ||
            estimateRun.isPending ||
            (est !== null && totalOut === 0)
          }
          onClick={() => create.mutate()}
        >
          Create tiled dataset
        </Button>

        <Text size="xs" c="dimmed">
          Tiles land unassigned in the new dataset — split them there.
        </Text>
      </Stack>
    </Card>
  )
}
