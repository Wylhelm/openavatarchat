import base64js from 'base64-js'
import { Buffer } from 'buffer'
import PythonStruct from 'python-struct'

// Websocket binary frames are Blobs in browsers, but some runtimes deliver an
// ArrayBuffer. ArrayBuffer.slice() returns an ArrayBuffer, which has no
// arrayBuffer() method — that is the "slice(...).arrayBuffer is not a function"
// crash. Text frames (keepalive Ping) are not motion data at all.
export function ensureBlob(data: unknown): Blob | null {
  if (typeof data === 'string' || data == null) {
    return null
  }
  if (data instanceof Blob) {
    return data
  }
  if (data instanceof ArrayBuffer) {
    return new Blob([data])
  }
  if (ArrayBuffer.isView(data)) {
    return new Blob([
      new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
    ])
  }
  return null
}

export async function readBlobBytes(blob: Blob): Promise<ArrayBuffer> {
  if (typeof blob.arrayBuffer === 'function') {
    return blob.arrayBuffer()
  }
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as ArrayBuffer)
    reader.onerror = () =>
      reject(reader.error ?? new Error('Failed to read motion blob'))
    reader.readAsArrayBuffer(blob)
  })
}

async function readText(blob: Blob): Promise<string> {
  if (typeof blob.text === 'function') {
    return blob.text()
  }
  const buffer = await readBlobBytes(blob)
  return new TextDecoder().decode(buffer)
}

export const unpack = async function (data: unknown, str = '<II') {
  const blob = ensureBlob(data)
  if (!blob) {
    throw new Error(
      `motion payload is ${Object.prototype.toString.call(data)}, not binary`,
    )
  }
  const unpackBuffer = await readBlobBytes(blob.slice(4, 12))
  const [jsonSize, binSize] = PythonStruct.unpack(
    str,
    Buffer.from(unpackBuffer),
  ) as number[]
  const jsonBlob = await readText(blob.slice(12, 12 + jsonSize))
  const parsedData = JSON.parse(jsonBlob)
  return {
    parsedData,
    jsonSize,
    binSize,
  }
}
export const mergeBlob = (strArray: string[], target: Uint8Array) => {
  let offset = 0
  strArray.forEach((str) => {
    const byteArray = base64js.toByteArray(str)
    target.set(byteArray, offset)
    offset += byteArray.byteLength
  })
  const blob = new Blob([target])
  return blob
}
