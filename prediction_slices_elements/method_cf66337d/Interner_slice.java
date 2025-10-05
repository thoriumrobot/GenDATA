// Source-based slice around line 50
// Method: <com.google.common.collect.Interner: E intern(E)>

   * both calls will return the same instance. That is, {@code intern(a).equals(a)} always holds,
   * and {@code intern(a) == intern(b)} if and only if {@code a.equals(b)}. Note that {@code
   * intern(a)} is permitted to return one instance now and a different instance later if the
   * original interned instance was garbage-collected.
   *
   * <p><b>Warning:</b> do not use with mutable objects.
   *
   * @throws NullPointerException if {@code sample} is null
   */
  E intern(E sample);
}
