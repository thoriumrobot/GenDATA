// Source-based slice around line 337
// Method: <com.google.common.collect.DiscreteDomain: C maxValue()>

   * input of type {@code C}.
   *
   * <p>The default implementation throws {@code NoSuchElementException}.
   *
   * @return the maximum value of type {@code C}; never null
   * @throws NoSuchElementException if the type has no (practical) maximum value; for example,
   *     {@link java.math.BigInteger}
   */
  @CanIgnoreReturnValue
  public C maxValue() {
    throw new NoSuchElementException();
  }
}
