// Source-based slice around line 56
// Method: <com.google.common.io.CharSequenceReader: boolean hasRemaining()>

    this.seq = checkNotNull(seq);
  }

  private void checkOpen() throws IOException {
    if (seq == null) {
      throw new IOException("reader closed");
    }
  }

  private boolean hasRemaining() {
    return remaining() > 0;
  }

  private int remaining() {
    requireNonNull(seq); // safe as long as we call this only after checkOpen
    return seq.length() - pos;
  }

  /*
   * To avoid the need to call requireNonNull so much, we could consider more clever approaches,
