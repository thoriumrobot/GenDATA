// Source-based slice around line 471
// Method: <com.google.common.reflect.TypeToken: boolean isSupertypeOf(Type)>


  /**
   * Returns true if this type is a supertype of the given {@code type}. "Supertype" is defined
   * according to <a
   * href="http://docs.oracle.com/javase/specs/jls/se8/html/jls-4.html#jls-4.5.1">the rules for type
   * arguments</a> introduced with Java generics.
   *
   * @since 19.0
   */
  public final boolean isSupertypeOf(Type type) {
    return of(type).isSubtypeOf(getType());
  }

  /**
   * Returns true if this type is a subtype of the given {@code type}. "Subtype" is defined
   * according to <a
   * href="http://docs.oracle.com/javase/specs/jls/se8/html/jls-4.html#jls-4.5.1">the rules for type
   * arguments</a> introduced with Java generics.
   *
   * @since 19.0
