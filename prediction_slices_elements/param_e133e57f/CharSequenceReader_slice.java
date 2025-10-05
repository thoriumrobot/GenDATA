// Source-based slice around line 115
// Method: <com.google.common.io.CharSequenceReader: long skip(long)>

    }
    int charsToRead = min(len, remaining());
    for (int i = 0; i < charsToRead; i++) {
      cbuf[off + i] = seq.charAt(pos++);
    }
    return charsToRead;
  }

  @Override
  public synchronized long skip(long n) throws IOException {
    checkArgument(n >= 0, "n (%s) may not be negative", n);
    checkOpen();
    int charsToSkip = (int) min(remaining(), n); // safe because remaining is an int
    pos += charsToSkip;
    return charsToSkip;
  }

  @Override
  public synchronized boolean ready() throws IOException {
    checkOpen();
