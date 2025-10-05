// Source-based slice around line 93
// Method: <com.google.common.io.CharSequenceReader: int read()>

    }
    int charsToRead = min(target.remaining(), remaining());
    for (int i = 0; i < charsToRead; i++) {
      target.put(seq.charAt(pos++));
    }
    return charsToRead;
  }

  @Override
  public synchronized int read() throws IOException {
    checkOpen();
    requireNonNull(seq); // safe because of checkOpen
    return hasRemaining() ? seq.charAt(pos++) : -1;
  }

  @Override
  public synchronized int read(char[] cbuf, int off, int len) throws IOException {
    checkPositionIndexes(off, off + len, cbuf.length);
    checkOpen();
    requireNonNull(seq); // safe because of checkOpen
