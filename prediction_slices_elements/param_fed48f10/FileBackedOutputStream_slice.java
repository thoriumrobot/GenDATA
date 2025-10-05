// Source-based slice around line 213
// Method: <com.google.common.io.FileBackedOutputStream: void write(byte[],int,int)>

    out.write(b);
  }

  @Override
  public synchronized void write(byte[] b) throws IOException {
    write(b, 0, b.length);
  }

  @Override
  public synchronized void write(byte[] b, int off, int len) throws IOException {
    update(len);
    out.write(b, off, len);
  }

  @Override
  public synchronized void close() throws IOException {
    out.close();
  }

  @Override
