// Source-based slice around line 98
// Method: <com.google.common.io.ByteSink: void write(byte[])>

        ? (BufferedOutputStream) out
        : new BufferedOutputStream(out);
  }

  /**
   * Writes all the given bytes to this sink.
   *
   * @throws IOException if an I/O occurs while writing to this sink
   */
  public void write(byte[] bytes) throws IOException {
    checkNotNull(bytes);

    try (OutputStream out = openStream()) {
      out.write(bytes);
    }
  }

  /**
   * Writes all the bytes from the given {@code InputStream} to this sink. Does not close {@code
   * input}.
