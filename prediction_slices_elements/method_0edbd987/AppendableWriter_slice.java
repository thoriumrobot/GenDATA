// Source-based slice around line 124
// Method: <com.google.common.io.AppendableWriter: void checkNotClosed()>

  }

  @Override
  public Writer append(@Nullable CharSequence charSeq, int start, int end) throws IOException {
    checkNotClosed();
    target.append(charSeq, start, end);
    return this;
  }

  private void checkNotClosed() throws IOException {
    if (closed) {
      throw new IOException("Cannot write to a closed writer.");
    }
  }
}
