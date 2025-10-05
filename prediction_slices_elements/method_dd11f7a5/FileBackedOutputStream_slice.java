// Source-based slice around line 161
// Method: <com.google.common.io.FileBackedOutputStream: ByteSource asByteSource()>

          };
    }
  }

  /**
   * Returns a readable {@link ByteSource} view of the data that has been written to this stream.
   *
   * @since 15.0
   */
  public ByteSource asByteSource() {
    return source;
  }

  private synchronized InputStream openInputStream() throws IOException {
    if (file != null) {
      return new FileInputStream(file);
    } else {
      // requireNonNull is safe because we always have either `file` or `memory`.
      requireNonNull(memory);
      return new ByteArrayInputStream(memory.getBuffer(), 0, memory.getCount());
