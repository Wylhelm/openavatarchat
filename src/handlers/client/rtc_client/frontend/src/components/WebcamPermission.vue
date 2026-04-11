<script setup lang="ts">
import { useVideoChatStore } from '@/store';
import { useVisionStore } from '@/store/vision';
import { VideoCameraOutlined } from '@ant-design/icons-vue';
import { onMounted } from 'vue';

const emits = defineEmits(['click']);
const videoChatState = useVideoChatStore();
const visionState = useVisionStore();
const accessClick = async () => {
  videoChatState.accessDevice();
};
onMounted(() => {
  // accessClick(); //自动获取权限
});

const text = 'Cliquer pour autoriser la cam\u00e9ra et le micro';
</script>

<template>
  <div class="access-wrap" @click="accessClick">
    <span class="icon-wrap">
      <VideoCameraOutlined />
    </span>
    {{ text }}
  </div>
</template>
<style lang="less" scoped>
.access-wrap {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.icon-wrap {
  width: 30px;
  font-size: 40px;
}
</style>
