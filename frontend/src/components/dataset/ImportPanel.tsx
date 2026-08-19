// 데이터셋으로 가져오기 — 두 방식.
//
//   동영상        프레임을 뽑아 넣는다. 영상은 서버가 끝나고 버린다
//   기존 데이터셋  YOLO zip 의 이미지·라벨을 넣고 클래스를 이 데이터셋에 병합한다
//
// 둘 다 결과는 **미검수**로 들어온다. 검수는 그 다음 일이다.
import { useState } from 'react'
import { Card, Group, Stack, Text, ThemeIcon } from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconFileZip } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { importYoloZip } from '../../api/client'
import { useJobStore } from '../../stores/jobStore'
import VideoExtractCard from '../upload/VideoExtractCard'

interface Props {
  projectId: string
  datasetId: string
}

export default function ImportPanel({ projectId, datasetId }: Props) {
  const qc = useQueryClient()
  const startDatasetImport = useJobStore((s) => s.startDatasetImport)
  // 서버가 영상 워커를 하나만 돌린다 — 추출이 도는 동안에는 새로 못 시작한다
  const busy = useJobStore((s) =>
    Object.values(s.jobs).some((j) => j.kind === 'import' && j.status === 'running'),
  )
  const [zipPercent, setZipPercent] = useState<number | null>(null)

  // 영상 추출은 잡이라, 끝났을 때의 갱신은 전역 `JobIndicator` 가 맡는다.
  // 여기서 쓰는 건 그 자리에서 끝나는 zip 가져오기용이다.
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ['dataset', projectId, datasetId] })
    qc.invalidateQueries({ queryKey: ['datasets', projectId] })
  }

  const importZip = useMutation({
    mutationFn: (file: File) =>
      importYoloZip(projectId, datasetId, file, undefined, { onProgress: setZipPercent }),
    onSuccess: (r) => {
      notifications.show({
        message: `Imported ${r.images} images (${r.labeled} labeled, ${r.classes} classes)`,
        color: 'green',
      })
      refresh()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
    onSettled: () => setZipPercent(null),
  })

  return (
    <Stack gap="md">
      <VideoExtractCard
        busy={busy}
        hint="Frames land in this dataset — the video itself is discarded"
        onStart={(file, params) => startDatasetImport(projectId, datasetId, file, params)}
      />

      <Card withBorder radius="md" padding="lg">
        <Stack>
          <Group gap="xs">
            <ThemeIcon variant="light" size="lg" radius="md">
              <IconFileZip size={20} />
            </ThemeIcon>
            <div>
              <Text fw={600}>Existing dataset</Text>
              <Text size="xs" c="dimmed">
                A YOLO zip (images + labels + data.yaml). Its classes merge into this dataset.
              </Text>
            </div>
          </Group>

          <Dropzone
            onDrop={(files) => files[0] && importZip.mutate(files[0])}
            accept={['application/zip', 'application/x-zip-compressed']}
            multiple={false}
            loading={importZip.isPending}
            radius="md"
            p="sm"
          >
            <Group gap="xs" justify="center" style={{ pointerEvents: 'none' }} py="md">
              <IconFileZip size={20} stroke={1.5} />
              <Text size="sm">
                {zipPercent !== null && zipPercent < 100
                  ? `Uploading… ${zipPercent}%`
                  : importZip.isPending
                    ? 'Importing…'
                    : 'Drop a YOLO dataset zip here or click to browse'}
              </Text>
            </Group>
          </Dropzone>
        </Stack>
      </Card>
    </Stack>
  )
}
