// Source-based slice around line 112
// Method: com.google.common.base.internal.Finalizer.finalizableReferenceClassReference

      logger.log(
          Level.INFO,
          "Failed to clear thread local values inherited by reference finalizer thread.",
          t);
    }

    thread.start();
  }

  private final WeakReference<Class<?>> finalizableReferenceClassReference;
  private final PhantomReference<Object> frqReference;
  private final ReferenceQueue<Object> queue;

  // By preference, we will use the Thread constructor that has an `inheritThreadLocals` parameter.
  // But before Java 9, our only way not to inherit ThreadLocals is to zap them after the thread
  // is created, by accessing a private field.
  private static final @Nullable Constructor<Thread> bigThreadConstructor =
      getBigThreadConstructor();

  private static final @Nullable Field inheritableThreadLocals =
