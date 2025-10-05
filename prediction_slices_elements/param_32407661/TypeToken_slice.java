// Source-based slice around line 495
// Method: <com.google.common.reflect.TypeToken: boolean isSubtypeOf(Type)>


  /**
   * Returns true if this type is a subtype of the given {@code type}. "Subtype" is defined
   * according to <a
   * href="http://docs.oracle.com/javase/specs/jls/se8/html/jls-4.html#jls-4.5.1">the rules for type
   * arguments</a> introduced with Java generics.
   *
   * @since 19.0
   */
  public final boolean isSubtypeOf(Type supertype) {
    checkNotNull(supertype);
    if (supertype instanceof WildcardType) {
      // if 'supertype' is <? super Foo>, 'this' can be:
      // Foo, SubFoo, <? extends Foo>.
      // if 'supertype' is <? extends Foo>, nothing is a subtype.
      return any(((WildcardType) supertype).getLowerBounds()).isSupertypeOf(runtimeType);
    }
    // if 'this' is wildcard, it's a suptype of to 'supertype' if any of its "extends"
    // bounds is a subtype of 'supertype'.
    if (runtimeType instanceof WildcardType) {
