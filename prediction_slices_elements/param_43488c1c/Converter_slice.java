// Source-based slice around line 248
// Method: <com.google.common.base.Converter: A unsafeDoBackward(B)>

   * to Converter.convert). So maybe we don't want to think too hard about how to prevent our
   * checkers from issuing errors related to LegacyConverter, since it turns out that
   * LegacyConverter does violate the assumptions we make elsewhere.
   */

  private @Nullable B unsafeDoForward(@Nullable A a) {
    return doForward(uncheckedCastNullableTToT(a));
  }

  private @Nullable A unsafeDoBackward(@Nullable B b) {
    return doBackward(uncheckedCastNullableTToT(b));
  }

  /**
   * Returns an iterable that applies {@code convert} to each element of {@code fromIterable}. The
   * conversion is done lazily.
   *
   * <p>The returned iterable's iterator supports {@code remove()} if the input iterator does. After
   * a successful {@code remove()} call, {@code fromIterable} no longer contains the corresponding
   * element.
