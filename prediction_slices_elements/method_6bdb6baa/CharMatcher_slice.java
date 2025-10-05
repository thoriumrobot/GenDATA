// Source-based slice around line 917
// Method: <com.google.common.base.CharMatcher: boolean apply(Character)>

   *     instead.
   */
  @InlineMe(replacement = "this.matches(character)")
  @Deprecated
  @Override
  // We can't compatibly make this `final` now.
  @InlineMeValidationDisabled(
      "While apply() is not final, the inlining is still safe because all known overrides of"
          + " apply() call matches().")
  public boolean apply(Character character) {
    return matches(character);
  }

  /**
   * @deprecated Provided only to satisfy the {@link java.util.function.Predicate} interface; use
   *     {@link #matches} instead.
   * @since 21.0
   */
  @InlineMe(replacement = "this.matches(character)")
  @Deprecated
