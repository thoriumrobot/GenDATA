// Source-based slice around line 261
// Method: <com.google.common.base.FinalizableReferenceQueue: Class loadFinalizer(FinalizerLoader)>

      }
    }
  }

  /**
   * Iterates through the given loaders until it finds one that can load Finalizer.
   *
   * @return Finalizer.class
   */
  private static Class<?> loadFinalizer(FinalizerLoader... loaders) {
    for (FinalizerLoader loader : loaders) {
      Class<?> finalizer = loader.loadFinalizer();
      if (finalizer != null) {
        return finalizer;
      }
    }

    throw new AssertionError();
  }

