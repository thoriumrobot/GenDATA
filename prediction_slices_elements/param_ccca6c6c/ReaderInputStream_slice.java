// Source-based slice around line 242
// Method: <com.google.common.io.ReaderInputStream: void startDraining(boolean)>

  private static int availableCapacity(Buffer buffer) {
    return buffer.capacity() - buffer.limit();
  }

  /**
   * Flips the buffer output buffer so we can start reading bytes from it. If we are starting to
   * drain because there was overflow, and there aren't actually any characters to drain, then the
   * overflow must be due to a small output buffer.
   */
  private void startDraining(boolean overflow) {
    Java8Compatibility.flip(byteBuffer);
    if (overflow && byteBuffer.remaining() == 0) {
      byteBuffer = ByteBuffer.allocate(byteBuffer.capacity() * 2);
    } else {
      draining = true;
    }
  }

  /**
   * Copy as much of the byte buffer into the output array as possible, returning the (positive)
