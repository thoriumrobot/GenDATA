// Source-based slice around line 60
// Method: <com.google.common.collect.DiscreteDomain: DiscreteDomain integers()>


  /**
   * Returns the discrete domain for values of type {@code Integer}.
   *
   * <p>This method always returns the same object. That object is serializable; deserializing it
   * results in the same object too.
   *
   * @since 14.0 (since 10.0 as {@code DiscreteDomains.integers()})
   */
  public static DiscreteDomain<Integer> integers() {
    return IntegerDomain.INSTANCE;
  }

  private static final class IntegerDomain extends DiscreteDomain<Integer> implements Serializable {
    private static final IntegerDomain INSTANCE = new IntegerDomain();

    IntegerDomain() {
      super(true);
    }

