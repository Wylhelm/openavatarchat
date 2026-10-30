const aecParam = new URLSearchParams(window.location.search).get('aec')

export const isCoarsePointer =
  window.matchMedia?.('(pointer: coarse)').matches ?? false

// Mobile devices (iOS Safari/Chrome and Android) route page audio through the
// OS phone-call (VoiceProcessingIO) path when getUserMedia opens a mic with
// echoCancellation/noiseSuppression/autoGainControl enabled, which causes
// WebAudio output to crackle. Mobile therefore opens the mic in raw media mode
// (noEchoCancel = true) while keeping the mic track open for vocal barge-in;
// the 270 Hz high-pass on mobile speaker output + server-side VAD spectral
// filter prevent the avatar from self-interrupting without hardware AEC.
//
// EXPERIMENT (2026-10-30): hands-free speakerphone use caused a hard echo
// loopback (the avatar retranscribed its own replies). Re-enable hardware AEC
// on mobile by default to test whether the iOS crackle actually returns.
export const noEchoCancel =
  aecParam === '0' || (aecParam !== '1' && isCoarsePointer && false)

