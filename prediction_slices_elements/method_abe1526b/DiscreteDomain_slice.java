// Source-based slice around line 321
// Method: <com.google.common.collect.DiscreteDomain: C minValue()>

   * input of type {@code C}.
   *
   * <p>The default implementation throws {@code NoSuchElementException}.
   *
   * @return the minimum value of type {@code C}; never null
   * @throws NoSuchElementException if the type has no (practical) minimum value; for example,
   *     {@link java.math.BigInteger}
   */
  @CanIgnoreReturnValue
  public C minValue() {
    throw new NoSuchElementException();
  }

  /**
   * Returns the maximum value of type {@code C}, if it has one. The maximum value is the unique
   * value for which {@link Comparable#compareTo(Object)} never returns a negative value for any
   * input of type {@code C}.
   *
   * <p>The default implementation throws {@code NoSuchElementException}.
   *
