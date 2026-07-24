import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Box, Button, Group, Image, Loader, Modal, SimpleGrid, Text } from '@mantine/core'
import { IconCheck } from '@tabler/icons-react'
import { listImages, type ImageItem } from '../../api/client'

interface Props {
  projectId: string
  opened: boolean
  onClose: () => void
  onPick: (img: ImageItem) => void
  /** multi-select: collect several, then confirm with onPickMany */
  multiple?: boolean
  onPickMany?: (imgs: ImageItem[]) => void
}

export default function DatasetPicker({ projectId, opened, onClose, onPick, multiple, onPickMany }: Props) {
  const [chosen, setChosen] = useState<Record<string, ImageItem>>({})
  const query = useQuery({
    queryKey: ['test-picker', projectId],
    queryFn: () => listImages(projectId, { size: 120, sort: 'created', order: 'desc' }),
    enabled: opened,
  })

  function toggle(img: ImageItem) {
    if (!multiple) {
      onPick(img)
      return
    }
    setChosen((prev) => {
      const next = { ...prev }
      if (next[img.name]) delete next[img.name]
      else next[img.name] = img
      return next
    })
  }

  const chosenList = Object.values(chosen)

  return (
    <Modal opened={opened} onClose={onClose} title="데이터셋에서 이미지 선택" size="xl">
      {query.isLoading ? (
        <Loader />
      ) : (
        <>
          <SimpleGrid cols={{ base: 3, sm: 5, md: 6 }} spacing="xs">
            {(query.data?.items ?? []).map((img) => {
              const picked = !!chosen[img.name]
              return (
                <Box key={img.name} pos="relative" style={{ cursor: 'pointer' }} onClick={() => toggle(img)}>
                  <Image
                    src={img.thumb}
                    radius="sm"
                    style={{ aspectRatio: '1', objectFit: 'cover', outline: picked ? '2px solid var(--mantine-color-blue-5)' : 'none' }}
                  />
                  {picked && (
                    <Box pos="absolute" top={4} right={4} bg="blue.5" style={{ borderRadius: 999, display: 'flex', padding: 2 }}>
                      <IconCheck size={12} color="#fff" />
                    </Box>
                  )}
                </Box>
              )
            })}
          </SimpleGrid>
          {multiple && (
            <Group justify="space-between" mt="md">
              <Text size="sm" c="dimmed">{chosenList.length}장 선택됨</Text>
              <Button disabled={!chosenList.length} onClick={() => { onPickMany?.(chosenList); setChosen({}) }}>
                선택 완료
              </Button>
            </Group>
          )}
        </>
      )}
    </Modal>
  )
}
