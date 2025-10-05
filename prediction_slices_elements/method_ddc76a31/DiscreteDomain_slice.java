// Source-based slice around line 199
// Method: <com.google.common.collect.DiscreteDomain: DiscreteDomain bigIntegers()>


  /**
   * Returns the discrete domain for values of type {@code BigInteger}.
   *
   * <p>This method always returns the same object. That object is serializable; deserializing it
   * results in the same object too.
   *
   * @since 15.0
   */
  public static DiscreteDomain<BigInteger> bigIntegers() {
    return BigIntegerDomain.INSTANCE;
  }

  private static final class BigIntegerDomain extends DiscreteDomain<BigInteger>
      implements Serializable {
    private static final BigIntegerDomain INSTANCE = new BigIntegerDomain();

    BigIntegerDomain() {
      super(true);
    }
