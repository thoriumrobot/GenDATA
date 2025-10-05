// Source-based slice around line 208
// Method: <com.google.common.io.FileBackedOutputStream: void write(byte[])>

  }

  @Override
  public synchronized void write(int b) throws IOException {
    update(1);
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
