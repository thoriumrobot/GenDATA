// Source-based slice around line 581
// Method: com.google.common.math.Stats.BYTES

        mean = calculateNewMeanNonFinite(mean, value);
      }
    }
    return mean;
  }

  // Serialization helpers

  /** The size of byte array representation in bytes. */
  static final int BYTES = (Long.SIZE + Double.SIZE * 4) / Byte.SIZE;

  /**
   * Gets a byte array representation of this instance.
   *
   * <p><b>Note:</b> No guarantees are made regarding stability of the representation between
   * versions.
   */
  public byte[] toByteArray() {
    ByteBuffer buff = ByteBuffer.allocate(BYTES).order(ByteOrder.LITTLE_ENDIAN);
    writeTo(buff);
