// Source-based slice around line 318
// Method: com.google.common.math.PairedStats.serialVersionUID

        BYTES,
        byteArray.length);
    ByteBuffer buffer = ByteBuffer.wrap(byteArray).order(ByteOrder.LITTLE_ENDIAN);
    Stats xStats = Stats.readFrom(buffer);
    Stats yStats = Stats.readFrom(buffer);
    double sumOfProductsOfDeltas = buffer.getDouble();
    return new PairedStats(xStats, yStats, sumOfProductsOfDeltas);
  }

  private static final long serialVersionUID = 0;
}
