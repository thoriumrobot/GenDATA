// Source-based slice around line 142
// Method: <com.google.common.io.CharSequenceReader: void reset()>


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

  @Override
  public synchronized void close() throws IOException {
    seq = null;
  }
}
