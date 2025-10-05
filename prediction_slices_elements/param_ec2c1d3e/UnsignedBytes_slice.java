// Source-based slice around line 509
// Method: <com.google.common.primitives.UnsignedBytes: byte flip(byte)>

    @Override
    // We use the class only after confirming that Arrays.compareUnsigned is available at runtime.
    @SuppressWarnings("Java8ApiChecker")
    @IgnoreJRERequirement
    public int compare(byte[] left, byte[] right) {
      return Arrays.compareUnsigned(left, right);
    }
  }

  private static byte flip(byte b) {
    return (byte) (b ^ 0x80);
  }

  /**
   * Sorts the array, treating its elements as unsigned bytes.
   *
   * @since 23.1
   */
  public static void sort(byte[] array) {
    checkNotNull(array);
