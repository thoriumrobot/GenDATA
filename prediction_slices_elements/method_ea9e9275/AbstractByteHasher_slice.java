// Source-based slice around line 79
// Method: <com.google.common.hash.AbstractByteHasher: Hasher putByte(byte)>

      update(scratch.array(), 0, bytes);
    } finally {
      Java8Compatibility.clear(scratch);
    }
    return this;
  }

  @Override
  @CanIgnoreReturnValue
  public Hasher putByte(byte b) {
    update(b);
    return this;
  }

  @Override
  @CanIgnoreReturnValue
  public Hasher putBytes(byte[] bytes) {
    checkNotNull(bytes);
    update(bytes);
    return this;
