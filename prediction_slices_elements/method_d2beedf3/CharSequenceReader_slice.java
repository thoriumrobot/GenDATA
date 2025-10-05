// Source-based slice around line 130
// Method: <com.google.common.io.CharSequenceReader: boolean markSupported()>

  }

  @Override
  public synchronized boolean ready() throws IOException {
    checkOpen();
    return true;
  }

  @Override
  public boolean markSupported() {
    return true;
  }

  @Override
  public synchronized void mark(int readAheadLimit) throws IOException {
    checkArgument(readAheadLimit >= 0, "readAheadLimit (%s) may not be negative", readAheadLimit);
    checkOpen();
    mark = pos;
  }

