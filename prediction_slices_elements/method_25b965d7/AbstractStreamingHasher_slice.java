// Source-based slice around line 187
// Method: <com.google.common.hash.AbstractStreamingHasher: HashCode hash()>

  @Override
  @CanIgnoreReturnValue
  public final Hasher putLong(long l) {
    buffer.putLong(l);
    munchIfFull();
    return this;
  }

  @Override
  public final HashCode hash() {
    munch();
    Java8Compatibility.flip(buffer);
    if (buffer.remaining() > 0) {
      processRemaining(buffer);
      Java8Compatibility.position(buffer, buffer.limit());
    }
    return makeHash();
  }

  /**
