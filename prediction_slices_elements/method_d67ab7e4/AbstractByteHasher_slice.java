// Source-based slice around line 68
// Method: <com.google.common.hash.AbstractByteHasher: Hasher update(ByteBuffer,int)>

      for (int remaining = b.remaining(); remaining > 0; remaining--) {
        update(b.get());
      }
    }
  }

  /** Updates the sink with the given number of bytes from the buffer. */
  @SuppressWarnings("ByteBufferBackingArray") // We created the array with ByteBuffer.allocate().
  @CanIgnoreReturnValue
  private Hasher update(ByteBuffer scratch, int bytes) {
    try {
      update(scratch.array(), 0, bytes);
    } finally {
      Java8Compatibility.clear(scratch);
    }
    return this;
  }

  @Override
  @CanIgnoreReturnValue
