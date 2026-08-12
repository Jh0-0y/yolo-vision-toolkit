import {
  Card,
  Image,
  SimpleGrid,
  Stack,
  Text,
} from '@mantine/core'
import {
  PLOT_LABELS,
} from './metrics'


export default function PlotsSection({
  files,
  onOpen,
}: {
  files: { name: string; url: string }[]
  onOpen: (v: { url: string; label: string }) => void
}) {
  const images = files.filter((f) => /\.(png|jpe?g)$/i.test(f.name))
  const key: { file: { name: string; url: string }; label: string }[] = []
  const preds: { name: string; url: string }[] = []
  for (const f of images) {
    if (/val_batch\d+_pred/i.test(f.name)) {
      preds.push(f)
      continue
    }
    const hit = PLOT_LABELS.find((p) => p.match.test(f.name))
    if (hit) key.push({ file: f, label: hit.label })
  }
  if (!key.length && !preds.length) return null
  return (
    <Stack gap="xs">
      {!!key.length && (
        <>
          <Text fw={600} size="sm">
            Results
          </Text>
          <SimpleGrid cols={{ base: 2, md: 4 }}>
            {key.map(({ file, label }) => (
              <Card
                key={file.name}
                withBorder
                p="xs"
                radius="md"
                style={{ cursor: 'pointer' }}
                onClick={() => onOpen({ url: file.url, label })}
              >
                <Image src={file.url} radius="sm" loading="lazy" h={140} fit="contain" />
                <Text size="xs" fw={500} mt={4} ta="center">
                  {label}
                </Text>
              </Card>
            ))}
          </SimpleGrid>
        </>
      )}
      {!!preds.length && (
        <>
          <Text fw={600} size="sm" mt="xs">
            Prediction samples
          </Text>
          <SimpleGrid cols={{ base: 2, md: 3 }}>
            {preds.map((f) => (
              <Card
                key={f.name}
                withBorder
                p="xs"
                radius="md"
                style={{ cursor: 'pointer' }}
                onClick={() => onOpen({ url: f.url, label: f.name })}
              >
                <Image src={f.url} radius="sm" loading="lazy" />
              </Card>
            ))}
          </SimpleGrid>
        </>
      )}
    </Stack>
  )
}

