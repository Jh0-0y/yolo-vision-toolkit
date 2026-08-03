import { useState } from 'react'
import {
  Card,
  Group,
  Stack,
  Tabs,
  Text,
  ThemeIcon,
  Title,
} from '@mantine/core'
import { Dropzone } from '@mantine/dropzone'
import { IconFileZip, IconMovie } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'
import VideoExtractCard from '../components/upload/VideoExtractCard'
import TilingOptions, { DEFAULT_TILING, type TilingState } from '../components/upload/TilingOptions'

export default function UploadPage() {
  const { projectId } = useParams() as { projectId: string }
  const queryClient = useQueryClient()
  const [tiling, setTiling] = useState<TilingState>(DEFAULT_TILING)

  const importZip = useMutation({
    mutationFn: (file: File) => {
      const form = new FormData()
      form.append('file', file)
      if (tiling.tile) {
        form.append('tiling', 'true')
        form.append('tile_size', String(tiling.tileSize))
        form.append('stride', String(tiling.stride))
        form.append('min_visibility', String(tiling.minVisibility))
        form.append('drop_empty', String(tiling.dropEmpty))
      }
      return api.upload<{ images: number; labeled: number; classes: number }>(
        `/projects/${projectId}/dataset-zip`,
        form,
      )
    },
    onSuccess: (res) => {
      notifications.show({
        message: `${res.images} images imported (${res.labeled} labeled, ${res.classes} classes)`,
        color: 'green',
      })
      queryClient.invalidateQueries({ queryKey: ['images', projectId] })
      queryClient.invalidateQueries({ queryKey: ['stats', projectId] })
    },
    onError: (e) => notifications.show({ message: String(e), color: 'red' }),
  })

  return (
    <Stack gap="lg">
      <div>
        <Title order={3}>Upload Data</Title>
        <Text c="dimmed" size="sm">
          Extract frames from a video, or import a labeled YOLO dataset — optionally tiled for training.
        </Text>
      </div>

      <Tabs defaultValue="video">
        <Tabs.List>
          <Tabs.Tab value="video" leftSection={<IconMovie size={16} />}>
            Video
          </Tabs.Tab>
          <Tabs.Tab value="dataset" leftSection={<IconFileZip size={16} />}>
            Labeled .zip
          </Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="video" pt="md">
          <VideoExtractCard projectId={projectId} />
        </Tabs.Panel>

        <Tabs.Panel value="dataset" pt="md">
          <Card withBorder radius="md" padding="lg">
            <Stack>
              <Group gap="xs">
                <ThemeIcon variant="light" size="lg" radius="md">
                  <IconFileZip size={20} />
                </ThemeIcon>
                <div>
                  <Text fw={600}>Labeled dataset (YOLO .zip)</Text>
                  <Text size="xs" c="dimmed">
                    Import an already-labeled dataset (images + labels + data.yaml)
                  </Text>
                </div>
              </Group>

              <TilingOptions
                value={tiling}
                onChange={setTiling}
                disabled={importZip.isPending}
                withLabels
              />

              <Dropzone
                onDrop={(files) => files[0] && importZip.mutate(files[0])}
                accept={['application/zip', 'application/x-zip-compressed']}
                multiple={false}
                loading={importZip.isPending}
                radius="md"
              >
                <Stack align="center" gap={6} py="xl" style={{ pointerEvents: 'none' }}>
                  <IconFileZip size={32} stroke={1.4} />
                  <Text size="sm">Drag a YOLO dataset .zip here or click to browse</Text>
                  <Text size="xs" c="dimmed">
                    images + labels + data.yaml — classes are merged in
                  </Text>
                </Stack>
              </Dropzone>

              {importZip.isPending && (
                <Text size="xs" c="dimmed">
                  Importing dataset…
                </Text>
              )}
            </Stack>
          </Card>
        </Tabs.Panel>
      </Tabs>
    </Stack>
  )
}
