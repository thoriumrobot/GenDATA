// Source-based slice around line 111
// Method: <com.google.common.io.AppendableWriter: Writer append(CharSequence)>


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
  }

  @Override
  public Writer append(@Nullable CharSequence charSeq, int start, int end) throws IOException {
    checkNotClosed();
    target.append(charSeq, start, end);
    return this;
