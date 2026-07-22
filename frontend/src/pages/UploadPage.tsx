import { useState } from 'react'
import {
  Card,
  Group,
  SimpleGrid,
  Stack,
  Text,
  ThemeIcon,
  Title,
} from '@mantine/core'
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone'
import { IconPhoto, IconUpload, IconX } from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'
import VideoExtractCard from '../components/upload/VideoExtractCard'

export default function UploadPage() {
  const { projectId } = useParams() as { projectId: string }
  const queryClient = useQueryClient()
  const [count, setCount] = useState(0)

  const upload = useMutation({
    mutationFn: (files: File[]) => {
      const form = new FormData()
      files.forEach((f) => form.append('files', f))
      setCount(files.length)
      return api.upload<{ added: number; skipped: number }>(
        `/projects/${projectId}/images`,
        form,
      )
    },
    onSuccess: (res) => {
      notifications.show({
        message: `${res.added} images added${res.skipped ? `, ${res.skipped} skipped` : ''}`,
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
          Upload images or extract frames from a video to build your dataset.
        </Text>
      </div>

      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
        <Card withBorder radius="md" padding="lg">
          <Stack>
            <Group gap="xs">
              <ThemeIcon variant="light" size="lg" radius="md">
                <IconPhoto size={20} />
              </ThemeIcon>
              <div>
                <Text fw={600}>Image upload</Text>
                <Text size="xs" c="dimmed">
                  Upload multiple images at once
                </Text>
              </div>
            </Group>

            <Dropzone
              onDrop={(files) => upload.mutate(files)}
              accept={IMAGE_MIME_TYPE}
              loading={upload.isPending}
              radius="md"
              style={{ flex: 1 }}
            >
              <Stack align="center" gap={6} py="xl" style={{ pointerEvents: 'none' }}>
                <Dropzone.Accept>
                  <IconUpload size={32} stroke={1.4} />
                </Dropzone.Accept>
                <Dropzone.Reject>
                  <IconX size={32} stroke={1.4} />
                </Dropzone.Reject>
                <Dropzone.Idle>
                  <IconPhoto size={32} stroke={1.4} />
                </Dropzone.Idle>
                <Text size="sm">Drag images here or click to browse</Text>
                <Text size="xs" c="dimmed">
                  jpg · png · bmp · webp · tiff
                </Text>
              </Stack>
            </Dropzone>

            {upload.isPending && (
              <Text size="xs" c="dimmed">
                Uploading {count} files…
              </Text>
            )}
          </Stack>
        </Card>

        <VideoExtractCard projectId={projectId} />
      </SimpleGrid>
    </Stack>
  )
}
