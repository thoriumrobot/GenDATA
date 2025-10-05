// Source-based slice around line 95
// Method: <com.google.common.io.FileBackedOutputStream: File getFile()>

    }

    int getCount() {
      return count;
    }
  }

  /** Returns the file holding the data (possibly null). */
  @VisibleForTesting
  synchronized @Nullable File getFile() {
    return file;
  }

  /**
   * Creates a new instance that uses the given file threshold, and does not reset the data when the
   * {@link ByteSource} returned by {@link #asByteSource} is finalized.
   *
   * @param fileThreshold the number of bytes before the stream should switch to buffering to a file
   * @throws IllegalArgumentException if {@code fileThreshold} is negative
   */
