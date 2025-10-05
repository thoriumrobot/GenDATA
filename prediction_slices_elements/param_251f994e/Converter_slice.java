// Source-based slice around line 505
// Method: <com.google.common.base.Converter: Converter from(Function,Function)>

   * <p>These functions will never be passed {@code null} and must not under any circumstances
   * return {@code null}. If a value cannot be converted, the function should throw an unchecked
   * exception (typically, but not necessarily, {@link IllegalArgumentException}).
   *
   * <p>The returned converter is serializable if both provided functions are.
   *
   * @since 17.0
   */
  public static <A, B> Converter<A, B> from(
      Function<? super A, ? extends B> forwardFunction,
      Function<? super B, ? extends A> backwardFunction) {
    return new FunctionBasedConverter<>(forwardFunction, backwardFunction);
  }

  private static final class FunctionBasedConverter<A, B> extends Converter<A, B>
      implements Serializable {
    private final Function<? super A, ? extends B> forwardFunction;
    private final Function<? super B, ? extends A> backwardFunction;

    private FunctionBasedConverter(
