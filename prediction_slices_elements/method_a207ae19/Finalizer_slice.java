// Source-based slice around line 232
// Method: <com.google.common.base.internal.Finalizer: Field getInheritableThreadLocalsField()>

      return null;
    }
    try {
      return finalizableReferenceClass.getMethod("finalizeReferent");
    } catch (NoSuchMethodException e) {
      throw new AssertionError(e);
    }
  }

  private static @Nullable Field getInheritableThreadLocalsField() {
    try {
      Field inheritableThreadLocals = Thread.class.getDeclaredField("inheritableThreadLocals");
      inheritableThreadLocals.setAccessible(true);
      return inheritableThreadLocals;
    } catch (Throwable t) {
      logger.log(
          Level.INFO,
          "Couldn't access Thread.inheritableThreadLocals. Reference finalizer threads will "
              + "inherit thread local values.");
      return null;
