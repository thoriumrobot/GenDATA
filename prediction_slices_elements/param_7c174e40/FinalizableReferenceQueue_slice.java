// Source-based slice around line 391
// Method: <com.google.common.base.FinalizableReferenceQueue: Method getStartFinalizer(Class)>

      try {
        return Class.forName(FINALIZER_CLASS_NAME);
      } catch (ClassNotFoundException e) {
        throw new AssertionError(e);
      }
    }
  }

  /** Looks up Finalizer.startFinalizer(). */
  static Method getStartFinalizer(Class<?> finalizer) {
    try {
      return finalizer.getMethod(
          "startFinalizer", Class.class, ReferenceQueue.class, PhantomReference.class);
    } catch (NoSuchMethodException e) {
      throw new AssertionError(e);
    }
  }
}
