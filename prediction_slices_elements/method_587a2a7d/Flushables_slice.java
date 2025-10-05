// Source-based slice around line 71
// Method: <com.google.common.io.Flushables: void flushQuietly(Flushable)>

  }

  /**
   * Equivalent to calling {@code flush(flushable, true)}, but with no {@code IOException} in the
   * signature.
   *
   * @param flushable the {@code Flushable} object to be flushed.
   */
  @Beta
  public static void flushQuietly(Flushable flushable) {
    try {
      flush(flushable, true);
    } catch (IOException e) {
      logger.log(Level.SEVERE, "IOException should not have been thrown.", e);
    }
  }
}
