// Source-based slice around line 220
// Method: <com.google.common.util.concurrent.ThreadFactoryBuilder: String format(String,Object)>

        }
        if (uncaughtExceptionHandler != null) {
          thread.setUncaughtExceptionHandler(uncaughtExceptionHandler);
        }
        return thread;
      }
    };
  }

  private static String format(String format, Object... args) {
    return String.format(Locale.ROOT, format, args);
  }
}
