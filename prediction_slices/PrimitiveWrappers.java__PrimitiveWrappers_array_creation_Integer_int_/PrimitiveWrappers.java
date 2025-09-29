import org.checkerframework.checker.index.qual.*;

public class PrimitiveWrappers {

    void array_creation(@NonNegative Integer i, @NonNegative int j) {
        int[] a = new int[j];
        int[] b = new int[i];
    }
}
