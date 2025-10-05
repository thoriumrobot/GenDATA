// Source-based slice around line 63
// Method: <com.google.common.base.MoreObjects: T firstNonNull(T,T)>

   * first.or(supplier)}.
   *
   * <p><b>Java 9 users:</b> use {@code java.util.Objects.requireNonNullElse(first, second)}
   * instead.
   *
   * @return {@code first} if it is non-null; otherwise {@code second} if it is non-null
   * @throws NullPointerException if both {@code first} and {@code second} are null
   * @since 18.0 (since 3.0 as {@code Objects.firstNonNull()}).
   */
  public static <T> T firstNonNull(@Nullable T first, @Nullable T second) {
    if (first != null) {
      return first;
    }
    if (second != null) {
      return second;
    }
    throw new NullPointerException("Both parameters are null");
  }

  /**
