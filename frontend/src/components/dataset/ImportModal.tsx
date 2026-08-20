// 데이터셋으로 가져오기 — 두 방식.
//
//   동영상        프레임을 뽑아 넣는다. 영상은 서버가 끝나고 버린다
//   기존 데이터셋  YOLO zip 의 이미지·라벨을 넣고 클래스를 이 데이터셋에 병합한다
//
// **모달이다.** 가져오기는 처음 한 번 하고 마는 일이라, 그리드 위에 늘 펼쳐 두면
// 정작 매일 보는 이미지가 아래로 밀린다.
import { useState } from 'react'
import { Card, Checkbox, Group, Modal, Stack, Tabs, Text, ThemeIcon } from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconFileZip, IconMovie } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { importYoloZip } from '../../api/client'
import { useJobStore } from '../../stores/jobStore'
import VideoExtractCard from '../upload/VideoExtractCard'

interface Props {
  projectId: string
  datasetId: string
  opened: boolean
  onClose: () => void
}

export default function ImportModal({ projectId, datasetId, opened, onClose }: Props) {
  const qc = useQueryClient()
  const startDatasetImport = useJobStore((s) => s.startDatasetImport)
  // 서버가 영상 워커를 하나만 돌린다 — 추출이 도는 동안에는 새로 못 시작한다
  const busy = useJobStore((s) =>
    Object.values(s.jobs).some((j) => j.kind === 'import' && j.status === 'running'),
  )
  const [zipPercent, setZipPercent] = useState<number | null>(null)
  // 사람이 이미 검증한 데이터인가. 기본은 참 — 남이 만든 YOLO 데이터셋을 가져오는
  // 것은 대개 완성품이다. 풀면 폴더 구조를 **읽지 않고** 전부 미검수로 들어온다.
  const [alreadyReviewed, setAlreadyReviewed] = useState(true)

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ['dataset', projectId, datasetId] })
    qc.invalidateQueries({ queryKey: ['datasets', projectId] })
    qc.invalidateQueries({ queryKey: ['dataset-images', projectId, datasetId] })
    qc.invalidateQueries({ queryKey: ['dataset-sources', projectId, datasetId] })
  }

  const importZip = useMutation({
    mutationFn: (file: File) =>
      importYoloZip(projectId, datasetId, file, alreadyReviewed, { onProgress: setZipPercent }),
    onSuccess: (r) => {
      notifications.show({
        message: r.reviewed
          ? `Imported ${r.images} images as reviewed — ${r.assigned} placed by folder`
          : `Imported ${r.images} images (${r.labeled} labeled, ${r.classes} classes)`,
        color: 'green',
      })
      refresh()
      onClose()
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
    onSettled: () => setZipPercent(null),
  })

  return (
    <Modal opened={opened} onClose={onClose} title="Import images" size="lg">
      <Tabs defaultValue="video" keepMounted={false}>
        <Tabs.List mb="md">
          <Tabs.Tab value="video" leftSection={<IconMovie size={16} />}>
            Video
          </Tabs.Tab>
          <Tabs.Tab value="zip" leftSection={<IconFileZip size={16} />}>
            Existing dataset
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="video">
          <VideoExtractCard
            busy={busy}
            onStart={(file, params) => {
              startDatasetImport(projectId, datasetId, file, params)
              onClose()
            }}
          />
        </Tabs.Panel>

        <Tabs.Panel value="zip">
          <Card withBorder radius="md" padding="lg">
            <Stack>
              <Group gap="xs">
                <ThemeIcon variant="light" size="lg" radius="md">
                  <IconFileZip size={20} />
                </ThemeIcon>
                <div>
                  <Text fw={600}>Existing dataset</Text>
                </div>
              </Group>

              <Checkbox
                checked={alreadyReviewed}
                onChange={(e) => setAlreadyReviewed(e.currentTarget.checked)}
                label="This data has already been reviewed"
              />

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
        </Tabs.Panel>
      </Tabs>
    </Modal>
  )
}
