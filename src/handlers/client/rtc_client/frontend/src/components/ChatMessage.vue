<script setup lang="ts">
const props = withDefaults(
  defineProps<{
    message: string,
    role: string,
    style?: string
  }>(),
  {}
)
</script>

<template>
  <div :class="['answer-message-container', role]" :style="style">
    <div class="answer-message-text">
      {{ message }}
    </div>
  </div>
</template>

<style scoped lang="less">
.answer-message-container {
  padding: 6px 12px;
  background: var(--bg-glass);
  border-radius: 12px;
  color: var(--text-primary);

  &.human {
    background: var(--message-human-bg);
    color: var(--message-human-text);
    margin-right: 0;
  }

  &.avatar {
    background: #9189fa;
    color: #ffffff;
  }
}
</style>
