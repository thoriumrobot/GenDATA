// Source-based slice around line 54
// Method: <com.google.common.hash.AbstractByteHasher: void update(ByteBuffer)>


  /** Updates this hasher with {@code len} bytes starting at {@code off} in the given buffer. */
  protected void update(byte[] b, int off, int len) {
    for (int i = off; i < off + len; i++) {
      update(b[i]);
    }
  }

  /** Updates this hasher with bytes from the given buffer. */
  protected void update(ByteBuffer b) {
    if (b.hasArray()) {
      update(b.array(), b.arrayOffset() + b.position(), b.remaining());
      Java8Compatibility.position(b, b.limit());
    } else {
      for (int remaining = b.remaining(); remaining > 0; remaining--) {
        update(b.get());
      }
    }
  }

