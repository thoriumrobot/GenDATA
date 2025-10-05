// Source-based slice around line 209
// Method: <com.google.common.base.Converter: A correctedDoBackward(B)>

  @Nullable B correctedDoForward(@Nullable A a) {
    if (handleNullAutomatically) {
      // TODO(kevinb): we shouldn't be checking for a null result at runtime. Assert?
      return a == null ? null : checkNotNull(doForward(a));
    } else {
      return unsafeDoForward(a);
    }
  }

  @Nullable A correctedDoBackward(@Nullable B b) {
    if (handleNullAutomatically) {
      // TODO(kevinb): we shouldn't be checking for a null result at runtime. Assert?
      return b == null ? null : checkNotNull(doBackward(b));
    } else {
      return unsafeDoBackward(b);
    }
  }

  /*
   * LegacyConverter violates the contract of Converter by allowing its doForward and doBackward
