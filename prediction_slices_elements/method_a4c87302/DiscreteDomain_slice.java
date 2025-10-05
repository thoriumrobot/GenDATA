// Source-based slice around line 124
// Method: <com.google.common.collect.DiscreteDomain: DiscreteDomain longs()>


  /**
   * Returns the discrete domain for values of type {@code Long}.
   *
   * <p>This method always returns the same object. That object is serializable; deserializing it
   * results in the same object too.
   *
   * @since 14.0 (since 10.0 as {@code DiscreteDomains.longs()})
   */
  public static DiscreteDomain<Long> longs() {
    return LongDomain.INSTANCE;
  }

  private static final class LongDomain extends DiscreteDomain<Long> implements Serializable {
    private static final LongDomain INSTANCE = new LongDomain();

    LongDomain() {
      super(true);
    }

