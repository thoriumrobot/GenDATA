// Source-based slice around line 450
// Method: <com.google.common.base.Converter: B apply(A)>

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0L;
  }

  /**
   * @deprecated Provided to satisfy the {@code Function} interface; use {@link #convert} instead.
   */
  @Deprecated
  @Override
  @InlineMe(replacement = "this.convert(a)")
  public final B apply(A a) {
    /*
     * Given that we declare this method as accepting and returning non-nullable values (because we
     * implement Function<A, B>, as discussed in a class-level comment), it would make some sense to
     * perform runtime null checks on the input and output. (That would also make NullPointerTester
     * happy!) However, since we didn't do that for many years, we're not about to start now.
     * (Runtime checks could be particularly bad for users of LegacyConverter.)
     *
     * Luckily, our nullness checker is smart enough to realize that `convert` has @PolyNull-like
     * behavior, so it knows that `convert(a)` returns a non-nullable value, and we don't need to
     * perform even a cast, much less a runtime check.
