  @CanonicalName String nonEmpty3(@FullyQualifiedName String s) {
    if (s.isEmpty()) {
      return null;
    } else {
      // :: error: (return)
      return s;
    }
  }
