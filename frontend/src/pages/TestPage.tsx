import { Card, Stack, Text, Title } from '@mantine/core'
import { IconFlask } from '@tabler/icons-react'

export default function TestPage() {
  return (
    <Card withBorder radius="md" padding="xl" mt="xl">
      <Stack align="center" gap="xs" py="xl">
        <IconFlask size={42} stroke={1.2} />
        <Title order={4}>Test — Coming soon</Title>
        <Text c="dimmed" size="sm" ta="center">
          Try your trained models on images right here — this page is under construction.
        </Text>
      </Stack>
    </Card>
  )
}
