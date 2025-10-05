// Source-based slice around line 52
// Method: <com.google.common.io.CountingOutputStream: void write(byte[],int,int)>

    super(checkNotNull(out));
  }

  /** Returns the number of bytes written. */
  public long getCount() {
    return count;
  }

  @Override
  public void write(byte[] b, int off, int len) throws IOException {
    out.write(b, off, len);
    count += len;
  }

  @Override
  public void write(int b) throws IOException {
    out.write(b);
    count++;
  }

