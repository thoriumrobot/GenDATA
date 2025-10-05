// Source-based slice around line 292
// Method: <com.google.common.hash.LittleEndianByteArray: LittleEndianBytes makeGetter()>

      }

      @Override
      public boolean usesFastPath() {
        return false;
      }
    }
  }

  static LittleEndianBytes makeGetter() {
    LittleEndianBytes usingVarHandle =
        VarHandleLittleEndianBytesMaker.INSTANCE.tryMakeVarHandleLittleEndianBytes();
    if (usingVarHandle != null) {
      return usingVarHandle;
    }

    try {
      /*
       * UnsafeByteArray uses Unsafe.getLong() in an unsupported way, which is known to cause
       * crashes on Android when running in 32-bit mode. For maximum safety, we shouldn't use
