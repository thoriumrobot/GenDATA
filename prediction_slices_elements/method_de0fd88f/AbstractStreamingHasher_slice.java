// Source-based slice around line 202
// Method: <com.google.common.hash.AbstractStreamingHasher: HashCode makeHash()>

    }
    return makeHash();
  }

  /**
   * Computes a hash code based on the data that have been provided to this hasher. This is called
   * after all chunks are handled with {@link #process} and any leftover bytes that did not make a
   * complete chunk are handled with {@link #processRemaining}.
   */
  protected abstract HashCode makeHash();

  // Process pent-up data in chunks
  private void munchIfFull() {
    if (buffer.remaining() < 8) {
      // buffer is full; not enough room for a primitive. We have at least one full chunk.
      munch();
    }
  }

  private void munch() {
