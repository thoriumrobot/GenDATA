// Source-based slice around line 135
// Method: <com.google.common.io.CharSequenceReader: void mark(int)>

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

  @Override
  public synchronized void reset() throws IOException {
    checkOpen();
    pos = mark;
  }
