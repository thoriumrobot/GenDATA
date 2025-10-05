// Source-based slice around line 662
// Method: com.google.common.math.Stats.serialVersionUID

        buffer.remaining());
    return new Stats(
        buffer.getLong(),
        buffer.getDouble(),
        buffer.getDouble(),
        buffer.getDouble(),
        buffer.getDouble());
  }

  private static final long serialVersionUID = 0;
}
