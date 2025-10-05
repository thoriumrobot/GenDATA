// Source-based slice around line 148
// Method: <com.google.common.io.CharSequenceReader: void close()>

  }

  @Override
  public synchronized void reset() throws IOException {
    checkOpen();
    pos = mark;
  }

  @Override
  public synchronized void close() throws IOException {
    seq = null;
  }
}
