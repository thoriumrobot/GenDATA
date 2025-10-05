// Source-based slice around line 302
// Method: <com.google.common.base.Converter: Converter reverse()>

  /**
   * Returns the reversed view of this converter, which converts {@code this.convert(a)} back to a
   * value roughly equivalent to {@code a}.
   *
   * <p>The returned converter is serializable if {@code this} converter is.
   *
   * <p><b>Note:</b> you should not override this method. It is non-final for legacy reasons.
   */
  @CheckReturnValue
  public Converter<B, A> reverse() {
    Converter<B, A> result = reverse;
    return (result == null) ? reverse = new ReverseConverter<>(this) : result;
  }

  private static final class ReverseConverter<A, B> extends Converter<B, A>
      implements Serializable {
    final Converter<A, B> original;

    ReverseConverter(Converter<A, B> original) {
      this.original = original;
