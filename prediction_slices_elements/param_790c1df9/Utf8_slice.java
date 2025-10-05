// Source-based slice around line 194
// Method: <com.google.common.base.Utf8: String unpairedSurrogateMsg(int)>

            || bytes[index++] > (byte) 0xBF
            // Fourth byte trailing-byte test
            || bytes[index++] > (byte) 0xBF) {
          return false;
        }
      }
    }
  }

  private static String unpairedSurrogateMsg(int i) {
    return "Unpaired surrogate at index " + i;
  }

  private Utf8() {}
}
