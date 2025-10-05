// Source-based slice around line 244
// Method: <com.google.common.base.Converter: B unsafeDoForward(A)>

   *
   * But no matter what we do, it's worth remembering that the resulting code is going to be unsound
   * in the presence of LegacyConverter, at least in the case of users who view the converter as a
   * Function<A, B> or who call convertAll (and for any checkers that apply @PolyNull-like semantics
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
