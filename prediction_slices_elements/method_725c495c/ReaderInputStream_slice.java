// Source-based slice around line 233
// Method: <com.google.common.io.ReaderInputStream: int availableCapacity(Buffer)>

    int numChars = reader.read(charBuffer.array(), limit, availableCapacity(charBuffer));
    if (numChars == -1) {
      endOfInput = true;
    } else {
      Java8Compatibility.limit(charBuffer, limit + numChars);
    }
  }

  /** Returns the number of elements between the limit and capacity. */
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
