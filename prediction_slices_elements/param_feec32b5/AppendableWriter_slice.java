// Source-based slice around line 104
// Method: <com.google.common.io.AppendableWriter: Writer append(char)>

  @Override
  public void close() throws IOException {
    this.closed = true;
    if (target instanceof Closeable) {
      ((Closeable) target).close();
    }
  }

  @Override
  public Writer append(char c) throws IOException {
    checkNotClosed();
    target.append(c);
    return this;
  }

  @Override
  public Writer append(@Nullable CharSequence charSeq) throws IOException {
    checkNotClosed();
    target.append(charSeq);
    return this;
