// Source-based slice around line 284
// Method: <com.google.common.collect.DiscreteDomain: C next(C)>


  /**
   * Returns the unique least value of type {@code C} that is greater than {@code value}, or {@code
   * null} if none exists. Inverse operation to {@link #previous}.
   *
   * @param value any value of type {@code C}
   * @return the least value greater than {@code value}, or {@code null} if {@code value} is {@code
   *     maxValue()}
   */
  public abstract @Nullable C next(C value);

  /**
   * Returns the unique greatest value of type {@code C} that is less than {@code value}, or {@code
   * null} if none exists. Inverse operation to {@link #next}.
   *
   * @param value any value of type {@code C}
   * @return the greatest value less than {@code value}, or {@code null} if {@code value} is {@code
   *     minValue()}
   */
  public abstract @Nullable C previous(C value);
