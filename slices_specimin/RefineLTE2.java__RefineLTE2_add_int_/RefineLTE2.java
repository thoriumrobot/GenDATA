import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.common.value.qual.MinLen;

public class RefineLTE2 {

    public void add(int elt) {
        if (num_values == values.length) {
            values = null;
            num_values++;
            return;
        }
        values[num_values] = elt;
        num_values++;
    }
}
