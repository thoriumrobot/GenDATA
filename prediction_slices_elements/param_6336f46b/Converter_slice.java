// Source-based slice around line 381
// Method: <com.google.common.base.Converter: Converter doAndThen(Converter)>

   *
   * <p>The returned converter is serializable if {@code this} converter and {@code secondConverter}
   * are.
   */
  public final <C> Converter<A, C> andThen(Converter<B, C> secondConverter) {
    return doAndThen(secondConverter);
  }

  /** Package-private non-final implementation of andThen() so only we can override it. */
  <C> Converter<A, C> doAndThen(Converter<B, C> secondConverter) {
    return new ConverterComposition<>(this, checkNotNull(secondConverter));
  }

  private static final class ConverterComposition<A, B, C> extends Converter<A, C>
      implements Serializable {
    final Converter<A, B> first;
    final Converter<B, C> second;

    ConverterComposition(Converter<A, B> first, Converter<B, C> second) {
      this.first = first;
