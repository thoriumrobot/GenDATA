// Source-based slice around line 270
// Method: <com.google.common.base.Converter: Iterable convertAll(Iterable)>

   * Just as Converter could implement `Function<@Nullable A, @Nullable B>` instead of `Function<A,
   * B>`, convertAll could accept and return iterables with nullable element types. In both cases,
   * we've chosen to instead use a signature that benefits existing users -- and is still safe.
   *
   * For convertAll, I haven't looked as closely at *how* much existing users benefit, so we should
   * keep an eye out for problems that new users encounter. Note also that convertAll could support
   * both use cases by using @PolyNull. (By contrast, we can't use @PolyNull for our superinterface
   * (`implements Function<@PolyNull A, @PolyNull B>`), at least as far as I know.)
   */
  public Iterable<B> convertAll(Iterable<? extends A> fromIterable) {
    checkNotNull(fromIterable, "fromIterable");
    return () ->
        new Iterator<B>() {
          private final Iterator<? extends A> fromIterator = fromIterable.iterator();

          @Override
          public boolean hasNext() {
            return fromIterator.hasNext();
          }

