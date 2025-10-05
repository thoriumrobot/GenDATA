// Source-based slice around line 933
// Method: <com.google.common.base.CharMatcher: boolean test(Character)>

   * @since 21.0
   */
  @InlineMe(replacement = "this.matches(character)")
  @Deprecated
  @Override
  // We can't compatibly make this `final` now.
  @InlineMeValidationDisabled(
      "While test() is not final, the inlining is still safe because all known overrides of test()"
          + " call matches().")
  public boolean test(Character character) {
    return matches(character);
  }

  /**
   * Returns a string representation of this {@code CharMatcher}, such as {@code
   * CharMatcher.or(WHITESPACE, JAVA_DIGIT)}.
   */
  @Override
  public String toString() {
    return super.toString();
