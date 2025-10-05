// Source-based slice around line 246
// Method: <com.google.common.base.internal.Finalizer: Constructor getBigThreadConstructor()>

    } catch (Throwable t) {
      logger.log(
          Level.INFO,
          "Couldn't access Thread.inheritableThreadLocals. Reference finalizer threads will "
              + "inherit thread local values.");
      return null;
    }
  }

  private static @Nullable Constructor<Thread> getBigThreadConstructor() {
    try {
      return Thread.class.getConstructor(
          ThreadGroup.class, Runnable.class, String.class, long.class, boolean.class);
    } catch (Throwable t) {
      // Probably pre Java 9. We'll fall back to Thread.inheritableThreadLocals.
      return null;
    }
  }
}
