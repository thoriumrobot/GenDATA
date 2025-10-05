// Source-based slice around line 91
// Method: <com.google.common.escape.Escaper: Function asFunction()>

   * @throws NullPointerException if {@code string} is null
   * @throws IllegalArgumentException if {@code string} contains badly formed UTF-16 or cannot be
   *     escaped for any other reason
   */
  public abstract String escape(String string);

  private final Function<String, String> asFunction = this::escape;

  /** Returns a {@link Function} that invokes {@link #escape(String)} on this escaper. */
  public final Function<String, String> asFunction() {
    return asFunction;
  }
}
