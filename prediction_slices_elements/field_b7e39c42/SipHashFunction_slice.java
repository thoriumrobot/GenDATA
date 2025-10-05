// Source-based slice around line 183
// Method: com.google.common.hash.SipHashFunction.serialVersionUID

        v1 = Long.rotateLeft(v1, 17);
        v3 = Long.rotateLeft(v3, 21);
        v1 ^= v2;
        v3 ^= v0;
        v2 = Long.rotateLeft(v2, 32);
      }
    }
  }

  private static final long serialVersionUID = 0L;
}
